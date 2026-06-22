import os
import urllib.request
import http
import socket
import logging
import time
import signal
import random
import queue
import threading
from collections import deque
import sqlite3

import config
from aki_model import Inference_Model, Feature_Processor
from messaging import Message_Parser
from database import SQLite_Manager
from metrics import (
    system_up, mllp_up, pager_up,
    messages_received_total, blood_tests_received_total,
    positive_predictions_total, negative_predictions_total,
    pager_failures_total, mllp_reconnections_total,
    creatinine_value_histogram, processing_latency_seconds,
    model_inferences_total, active_patients_gauge, start_metrics_server,
    ErrorMessageCounterHandler
)


class System_Engine:
    def __init__(self,
                 MLLP_HOST:str,
                 MLLP_PORT:int,
                 PAGER_URL:str, 
                 weights_filepath:str,
                 db_filepath:str='db',
                 schema_filepath:str='db/schema.sql',
                 history_filepath:str='data/history.csv',
                 system_logs_filepath:str='logs/system.log',
                 message_logs_path:str='logs/message.log'):
        """
        System Engine that opens a connection to receive HL7 messages
        - Parse them with Message_Parser from messaging.py
        - Calls the relevant feature extraction from model.py based on message type
        - Updates database accordingly
        - Also handles sending a POST request to the pager upon a positive AKI inference

        Args:
            MLLP_HOST (str): Host name of MLLP 
            MLLP_PORT (int): port number of MLLP
            PAGER_URL (str): url to pager
            weights_filepath (str): filepath for inference model weights
            db_filepath (str): database filepath
            history_filepath (str): history csv filepath for dataabase
            system_logs_filepath (str): logs filepath to log all system events
            message_logs_path (str): logs filepath which logs all received raw message and time message was processed
        """
        self.MLLP_HOST = MLLP_HOST
        self.MLLP_PORT = MLLP_PORT
        self.PAGER_URL = PAGER_URL

        # Create alert queue system
        self.alert_queue = queue.Queue()
        self.alert_worker = threading.Thread(target=self._alert_worker, daemon=True)
        # queue_data is shared between the socket thread (_send_alert) and the worker
        # thread (_clear_persisted_alert); guard every access with this lock.
        self.queue_data = {}
        self._queue_lock = threading.Lock()

        # Create loggers
        self.message_logger = self._create_logger('MESSAGES', message_logs_path)
        self.system_logger = self._create_logger('SYSTEM', system_logs_filepath)
        self.system_logger.addHandler(ErrorMessageCounterHandler())

        # Initialise other base classes
        self.message_parser = Message_Parser()
        self.sqlite_manager = SQLite_Manager(db_filepath)
        self.inference_model = Inference_Model(weights_filepath)
        self.system_logger.info(f"Printing the threshold {self.inference_model.threshold}")
        self.feature_processor = Feature_Processor()

        # Latency and metrics values
        start_metrics_server(port=8000)
        pager_up.set(1)  # optimistic until the first pager attempt updates it
        self.latency_window = deque(maxlen=1000)
        self.latency_count = 0
        self.latency_sum = 0.0
        self.latency_sum_sq = 0.0

        # SIGTERM shutdown. _sock holds the active MLLP socket so the signal handler can
        # close it to interrupt a blocking recv and shut down promptly.
        self._sock = None
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        
        # Initialise system
        self._initialise_system(history_filepath, schema_filepath)


    def _handle_sigterm(self, signum, frame):
        self.system_logger.info("Received SIGTERM, shutting down gracefully...")
        self.running = False
        # Interrupt a blocked socket read so the receive loop notices promptly.
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


    def _create_logger(self, name:str, logs_path:str):
        """
        Function to initialise logging 
        Sends all log files to message_logs_path
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        if not logger.handlers:
            os.makedirs(os.path.dirname(logs_path), exist_ok=True)
            handler = logging.FileHandler(logs_path)
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                "%m/%d/%Y %I:%M:%S %p"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger


    def _initialise_system(self, history_filepath:str, schema_filepath:str):
        """
        Function to preprocess history.csv to update the database
        - Also updates alert queue to send any past alerts that have not been sent

        Args:
            history_filepath (str): Filepath to history.csv
        """
        self.system_logger.info("Initialising AKI Inference Engine")
        self.system_logger.info(f"MLLP Host: {self.MLLP_HOST}, MLLP Port: {self.MLLP_PORT}")
        self.system_logger.info(f"PAGER_URL: {self.PAGER_URL}")

        # Connect to DB
        self.sqlite_manager._initialise_database()  

        # Check database records exists
        try:
            self.sqlite_manager.execute(
                "SELECT 1 FROM patients LIMIT 1;"
            )
            db_is_new = False
        except sqlite3.OperationalError:
            db_is_new = True

        if db_is_new:
            # Update schema and process history
            with open(schema_filepath, 'r') as f:
                schema_path = f.read()
            self.sqlite_manager.execute(schema_path)
            processed_history = self.feature_processor.process_history(history_filepath)
            self.sqlite_manager.create({"table": "patients", "values": processed_history})
            self.system_logger.info(f"Loaded {len(processed_history)} patient records from history")

        else:
            self.system_logger.info("Database already exists, skipping schema history load")

        # Update alert queue now
        unsent_alerts = self.sqlite_manager.fetchall("SELECT PID, To_Send_Positive_AKI_Msg_Timestamp FROM patients WHERE To_Send_Positive_AKI_Msg_Timestamp IS NOT NULL")
        for row in unsent_alerts:
            # fetchall returns a list of dicts - index by column name (do not unpack,
            # which would yield the column-name keys rather than the values).
            pid = str(row["PID"])
            timestamps = row["To_Send_Positive_AKI_Msg_Timestamp"].split(",")
            self.queue_data[pid] = set(timestamps)
            for ts in timestamps:
                self.alert_queue.put((pid, str(ts)))
                self.system_logger.info(f"Queued positive alert for PID:{pid}, timestamp:{ts}")

        # Start alert worker
        self.alert_worker.start()
        self.system_logger.info(f"Started alert worker, {self.alert_queue.qsize()} alerts in queue.")

        # Reflect how many patients we currently know about (for the dashboard gauge).
        patient_count = self.sqlite_manager.fetchone("SELECT COUNT(*) AS c FROM patients")
        active_patients_gauge.set(patient_count['c'] if patient_count else 0)


    def _alert_worker(self):
        """
        Drain the alert queue, delivering each alert to the pager.

        A persisted alert is only cleared from the DB once it has been delivered, so an
        undelivered alert (e.g. while shutting down) survives to be replayed on the next
        start rather than being silently dropped.
        """
        while self.running:
            try:
                PID, msg_time = self.alert_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if self._deliver_alert(PID, msg_time):
                    self._clear_persisted_alert(PID, msg_time)
            except Exception as e:
                self.system_logger.error(f"Alert worker unexpected error: {e}")

    def _deliver_alert(self, PID, msg_time):
        """
        POST one alert to the pager, retrying with exponential backoff (+ jitter) until it
        succeeds or the system is shutting down. Returns True iff the pager accepted it.
        """
        backoff = 1.0
        max_backoff = 30.0
        while self.running:
            try:
                data = f'{PID},{msg_time}'.encode("utf-8")
                r = urllib.request.urlopen(self.PAGER_URL, data=data, timeout=5)
                if r.status == http.HTTPStatus.OK:
                    self.system_logger.info(f"Alert Sent for {PID}, {msg_time}")
                    pager_up.set(1)
                    return True
                self.system_logger.error(f"Alert failed for {PID}, {msg_time}: status {r.status}")
                pager_failures_total.labels(reason='http_error').inc()
                pager_up.set(0)

            except urllib.error.URLError as e:
                self.system_logger.error(f"Alert failed for {PID}, {msg_time}: {str(e)}")
                pager_failures_total.labels(reason='connection_error').inc()
                pager_up.set(0)

            except Exception as e:
                self.system_logger.error(f"Alert failed for {PID}, {msg_time}: {str(e)}")
                pager_failures_total.labels(reason='unexpected').inc()
                pager_up.set(0)

            # Back off before retrying: a persistently failing pager (including repeated
            # non-200 responses) must not busy-loop, and concurrent retries should not
            # synchronise. Skip the wait if we are shutting down.
            if self.running:
                time.sleep(min(backoff, max_backoff) * (0.5 + random.random()))
                backoff = min(backoff * 2, max_backoff)
        return False

    def _clear_persisted_alert(self, PID, msg_time):
        """
        Remove a delivered alert from the in-memory set and the DB. Together with
        _send_alert this is the only place that mutates the shared alert state, so both
        take _queue_lock.
        """
        with self._queue_lock:
            pid_timestamps = self.queue_data.get(str(PID))
            if pid_timestamps is None:
                return
            pid_timestamps.discard(str(msg_time))
            if pid_timestamps:
                self.queue_data[str(PID)] = pid_timestamps
                remaining = ",".join(sorted(pid_timestamps))
                self.sqlite_manager.update({"table": "patients",
                                            "set": {"To_Send_Positive_AKI_Msg_Timestamp": remaining},
                                            "where": {"PID": int(PID)}})
            else:
                del self.queue_data[str(PID)]
                self.sqlite_manager.update({"table": "patients",
                                            "set": {"To_Send_Positive_AKI_Msg_Timestamp": None},
                                            "where": {"PID": int(PID)}})

    def _send_alert(self, PID:str, msg_time:str):
        """
        Function to send alert to the queue & update db

        Args:
            PID (str): PID of patient with positive aki inference
            msg_time (str): Datetime of the labortary results with the AKI
        """
        # Persist the unsent alert (under the lock) BEFORE enqueuing it, so a positive
        # prediction is durable the moment we ACK the source message - if we crash before
        # delivery, it is replayed from the DB on the next start.
        pid = str(PID)
        ts = str(msg_time)
        with self._queue_lock:
            timestamps = self.queue_data.get(pid, set())
            timestamps.add(ts)
            self.queue_data[pid] = timestamps
            persisted = ",".join(sorted(timestamps))
            self.sqlite_manager.update({"table": "patients",
                                        "set": {"To_Send_Positive_AKI_Msg_Timestamp": persisted},
                                        "where": {"PID": int(pid)}})
        self.alert_queue.put((pid, ts))


    def _admit_process(self, pid:str, parsed_message:dict):
        """
        Function detailing admit process for system

        Returns:
            error (bool): Boolean if any error occured and a resend of message is required
        """
        self.system_logger.info(f"Admit Message Received for Patient {pid}")
        error = False

        try:
            dob = parsed_message['PID'][7] 
            sex = parsed_message['PID'][8].lower()
            msg_timestamp = parsed_message['MSH'][6] #YYYYMMDDHHMMSS

            msg_year = int(msg_timestamp[:4])
            msg_month = int(msg_timestamp[4:6])
            msg_day = int(msg_timestamp[6:8])
            
            birth_year = int(dob[:4])
            birth_month = int(dob[4:6])
            birth_day = int(dob[6:8])

            # calculate age
            age = msg_year - birth_year
            if (msg_month, msg_day) < (birth_month, birth_day):
                age -= 1
            
            sex = 1 if sex == 'm' else 0

            patient_exists = self.sqlite_manager.fetchone(
                "SELECT PID FROM patients WHERE PID = ?",
                (int(pid),)
            )

            if patient_exists:
                # Patient exists - update age and sex
                self.sqlite_manager.update({
                    "table": "patients",
                    "set": {"Age": age, "Sex": sex},
                    "where": {"PID": int(pid)}
                })
                self.system_logger.info(f"Updated patient {pid}: Age={age}, Sex={sex}")
            else:
                # Patient does not exist - create new entry with age and sex
                self.sqlite_manager.create({
                    "table": "patients",
                    "values": {
                        "PID": int(pid),
                        "Age": age,
                        "Sex": sex
                    }
                })
                self.system_logger.info(f"Created new patient {pid}: Age={age}, Sex={sex}")
                active_patients_gauge.inc()
                
        except Exception as e:
            self.system_logger.error(f"Error processing admit message for patient {pid}: {str(e)}")
            error = True
        
        return error

    def _inference_process(self, pid:str, parsed_message:dict):
        """
        Function detailing inference process for a new laboratary result
        
        Returns:
            error (bool): Boolean if any error occured and a resend of message is required
            alert (bool): Boolean if a positive prediction was obtained
        """
        self.system_logger.info(f"Labotary Results Message Received for Patient {pid}")
        error = False
        alert = False # Currently set to true just to test for functionality, to update
        try:
            # Extract creatinine result from OBX segment
            # OBX.5 contains the observed value
            creatinine_result = float(parsed_message['OBX'][5])

            blood_tests_received_total.inc()
            creatinine_value_histogram.observe(creatinine_result)

            # Retrieve patient data from database
            patient_data = self.sqlite_manager.fetchone(
                "SELECT * FROM patients WHERE PID = ?",
                (int(pid),)
            )
            
            if not patient_data:
                self.system_logger.error(f"Patient {pid} not found in database for inference")
                error = True
                return error, alert
            
            # Check if patient has Age and Sex (required for inference)
            if patient_data['Age'] is None or patient_data['Sex'] is None:
                self.system_logger.error(f"Patient {pid} missing Age or Sex data")
                error = True
                return error, alert
            
            # Process features with new creatinine result
            updated_row, feature_array = self.feature_processor.process_features(
                patient_data, 
                creatinine_result
            )
            
            model_inferences_total.inc()

            has_aki = self.inference_model.inference(feature_array)

            if has_aki:
                alert = True
                self.system_logger.info(f"POSITIVE AKI prediction for patient {pid}")
            else:
                self.system_logger.info(f"Negative AKI prediction for patient {pid}")
            
            # Update database with new features
            # Remove PID from updated_row before update
            update_data = {k: v for k, v in updated_row.items() if k != 'PID'}
            self.sqlite_manager.update({
                "table": "patients",
                "set": update_data,
                "where": {"PID": int(pid)}
            })
            
        except Exception as e:
            self.system_logger.error(f"Error processing inference for patient {pid}: {str(e)}")
            error = True
        
        return error, alert


    def _discharge_process(self, pid:str, parsed_message:dict):
        """
        Function detailing discharge process for system, currently only logs data
        
        Returns:
            error (bool): Boolean if any error occured and a resend of message is required
        """
        self.system_logger.info(f"Discharge Message Received for Patient {pid}")
        return False


    def _route_message(self, parsed_message, raw_message, positive_predictions):
        """
        Dispatch one parsed message by type and queue an alert on a positive prediction.
        Returns (error, alert).
        """
        error = False
        alert = False
        pid = parsed_message['PID'][3]
        message_type = parsed_message['MSH'][8]

        if message_type == 'ADT^A01':
            messages_received_total.labels(message_type='admit').inc()
            error = self._admit_process(pid, parsed_message)
        elif message_type == 'ADT^A03':
            messages_received_total.labels(message_type='discharge').inc()
            error = self._discharge_process(pid, parsed_message)
        elif message_type == 'ORU^R01':
            messages_received_total.labels(message_type='blood_test').inc()
            error, alert = self._inference_process(pid, parsed_message)
        else:
            messages_received_total.labels(message_type='unknown').inc()
            self.system_logger.error(f"UNKNOWN Message Type Retrieved: {message_type}\nRaw Message:\n{raw_message}")
            error = True

        if alert:
            positive_predictions.append((int(parsed_message['PID'][3]), int(parsed_message['MSH'][6])))
            self._send_alert(str(parsed_message['PID'][3]), str(parsed_message['MSH'][6]))
            self.system_logger.info(f"Alert Queued: {parsed_message['PID'][3], parsed_message['MSH'][6]}")
            positive_predictions_total.inc()
        elif message_type == 'ORU^R01':
            negative_predictions_total.inc()

        return error, alert

    def _record_latency(self, msg_received_time):
        """Record end-to-end processing latency (distribution exported via the histogram)."""
        latency = time.perf_counter() - msg_received_time
        processing_latency_seconds.observe(latency)
        self.latency_window.append(latency)
        self.latency_count += 1
        self.latency_sum += latency
        self.latency_sum_sq += latency * latency
        # Per-message latency is high-frequency; keep it at DEBUG so it does not flood the
        # log over a multi-week run. The distribution is exported via the histogram.
        self.system_logger.debug(f"Latency for this msg from start to end: {latency:.6f}s")


    def receive_message(self):
        """
        System workflow function
        - Receives & Parse messages
        - Runs relevant pipeline base on message type
        - Pings Pager if necessary
        - Acknowledges message and wait for next one

        Returns list of tuples of all positive predictions made in the format of
        [(PID, time of HL7 message of positive laboratory result)]
        """
        positive_predictions = []

        idle_timeout = 60 * 60  # 60 minutes of no messages before we reset the connection
        reconnect_delay = 2

        # While system is running
        while self.running: 
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # Close connection if idle for 1 hour
                    # Allow us to constantly check if the simulator is functional
                    s.settimeout(idle_timeout)
                    s.connect((self.MLLP_HOST, self.MLLP_PORT))
                    self._sock = s
                    system_up.set(1)
                    mllp_up.set(1)
                    self.system_logger.info("Connected to messaging socket")

                    mllp_buffer = b""
                    while self.running:
                        # --- Transport: read bytes and reassemble complete MLLP frames ---
                        try:
                            buffer = s.recv(8192)
                        except socket.timeout:
                            # Idle for too long: drop and reconnect so a dead peer is noticed.
                            self.system_logger.info("No messages received for an hour, restarting messaging connection.")
                            break

                        # A zero-length read means the peer closed the connection. This is
                        # expected when the simulator restarts or supersedes our connection,
                        # so reconnect rather than treating it as the end of the stream.
                        if len(buffer) == 0:
                            self.system_logger.info("Message socket closed by server, reconnecting")
                            break

                        # A single read may hold a partial frame (e.g. a message split across
                        # TCP segments) or several; process only complete frames and carry the
                        # remainder over to the next read.
                        mllp_buffer += buffer
                        frames, mllp_buffer = self.message_parser.extract_frames(mllp_buffer)

                        for frame in frames:
                            msg_received_time = time.perf_counter()

                            # A malformed or unprocessable message must be NAKed (AE) so the
                            # sender resends it; it must never crash the loop.
                            error = False
                            try:
                                parsed_message, raw_message = self.message_parser.parse_mllp(frame)
                                self.message_logger.info(raw_message)
                                error, _ = self._route_message(parsed_message, raw_message, positive_predictions)
                            except Exception as e:
                                self.system_logger.error(f"Error processing message: {e}")
                                error = True

                            # Acknowledge: AA on success, AE on any processing error.
                            s.sendall(self.message_parser.format_response(error))
                            self.message_logger.info("Message Processed.")
                            self._record_latency(msg_received_time)
        
            # Connection-level failure: connect refused, reset by peer, broken pipe, etc.
            except (ConnectionResetError, ConnectionRefusedError, BrokenPipeError, OSError) as e:
                self.system_logger.warning(f"Messaging connection error: {e}")

            # We left the connection (peer close, idle timeout, or transport error).
            self._sock = None
            mllp_up.set(0)
            if not self.running:
                break
            mllp_reconnections_total.inc()
            self.system_logger.info(f"Reconnecting in {reconnect_delay}s...")
            time.sleep(reconnect_delay)

        system_up.set(0)
        # Return once the loop stops (e.g. after SIGTERM).
        return positive_predictions


    def shutdown(self):
        # Stop accepting work and interrupt any blocked socket read.
        self.running = False
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        # Let the alert worker finish its current delivery and exit before we close the
        # DB, so we never close the connection out from under a mid-flight write. Any
        # alert not delivered by now stays persisted and is replayed on the next start.
        if self.alert_worker.is_alive():
            self.alert_worker.join(timeout=15)

        system_up.set(0)
        mllp_up.set(0)
        try:
            self.sqlite_manager.close()
            self.system_logger.info("Database connection closed")
        except Exception as e:
            self.system_logger.error(f"Error during shutdown: {str(e)}")

        self.system_logger.info("Shutdown complete")


if __name__ == '__main__':
    # Generate directories
    os.makedirs('/state/logs', exist_ok=True)
    os.makedirs('/state/db', exist_ok=True)

    aki_system = System_Engine(
        MLLP_HOST=config.MLLP_HOST,
        MLLP_PORT=config.MLLP_PORT,
        PAGER_URL=config.PAGER_URL,
        weights_filepath=config.MODEL_PATH,
        db_filepath=config.DB_PATH,
        schema_filepath=config.SCHEMA_PATH,
        history_filepath=config.HISTORY_PATH,
        system_logs_filepath=config.SYSTEM_LOG_PATH,
        message_logs_path=config.MESSAGE_LOG_PATH
    )

    # Reconnect to live environment
    try:
        results = aki_system.receive_message()
    finally:
        aki_system.shutdown()