# Test overall integration flow
# Using mock history.csv & mock patient messages
# Using postive cases from the training data
# Mock message will be sent from a mock connection
# Boiler template from simulator_test.py
import http
import os
import signal
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
import pandas as pd
import numpy as np
from simulator.simulator_test import (TEST_MLLP_PORT, TEST_PAGER_PORT, wait_until_healthy, to_mllp)
from main import System_Engine
from metrics import positive_predictions_total


unittest.TestLoader.sortTestMethodsUsing = None


# Mock messages for admission, 2 labotary results (negative & postive), and discharge
ADT_A01 = [
    r"MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240729063000||ADT^A01|||2.5",
    r"PID|1||189386313||ELIZABETH HOLMES||19830203|F",
    r"NK1|1|SUNNY BALWANI|PARTNER"
]

ORU_R01 = [
    r"MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240730103000||ORU^R01|||2.5",
    r"PID|1||189386313",
    r"OBR|1||||||20240730094300",
    r"OBX|1|SN|CREATININE||93.95",
]

ORU_R02 = [
    r"MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240730125000||ORU^R01|||2.5",
    r"PID|1||189386313",
    r"OBR|1||||||20240730123900",
    r"OBX|1|SN|CREATININE||165.96",
]

ADT_A03 = [
    r"MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||202400805000||ADT^A03|||2.5",
    r"PID|1||189386313",
]

DATABASE_ADMISSION_DATA = [{
    'PID': 189386313, 
    'Age': 41,
    'Sex': 0,
    'Creatinine_Count': 6,
    'Max_Creatinine': 83.89,
    'Min_Creatinine': 61.41,
    'First_Creatinine':77.08,
    'Last_Creatinine': 83.89,
    'Creatinine_Mean': 76.055,
    'Creatinine_Std': 9.29857,
    'Creatinine_Change': 6.81,
    'Creatinine_CV': 0.12226,
    'Creatinine_Max_Change': 0.0,
    'Creatinine_Max_Consec_Increase': 22.25,
    'Creatinine_Max_Consec_Decrease':15.67,
    'Creatinine_Increase_Count': 3,
    'Creatinine_Count_Since_Peak': 0,
    'Creatinine_gt1p5_Count': 0,
    'Creatinine_Last_gt130_Count': 0,
    'To_Send_Positive_AKI_Msg_Timestamp': None
}]

DATABASE_LAB1_DATA = [{
    'PID': 189386313, 
    'Age': 41,
    'Sex': 0,
    'Creatinine_Count': 7,
    'Max_Creatinine': 93.95,
    'Min_Creatinine': 61.41,
    'First_Creatinine':77.08,
    'Last_Creatinine': 93.95,
    'Creatinine_Mean': 78.61143,
    'Creatinine_Std': 10.85358,
    'Creatinine_Change': 16.87,
    'Creatinine_CV': 0.13807,
    'Creatinine_Max_Change': 0.0,
    'Creatinine_Max_Consec_Increase': 22.25,
    'Creatinine_Max_Consec_Decrease':15.67,
    'Creatinine_Increase_Count': 4,
    'Creatinine_Count_Since_Peak': 0,
    'Creatinine_gt1p5_Count': 0,
    'Creatinine_Last_gt130_Count': 0,
    'To_Send_Positive_AKI_Msg_Timestamp': None
}]

DATABASE_LAB2_DATA = [{
    'PID': 189386313, 
    'Age': 41,
    'Sex': 0,
    'Creatinine_Count': 8,
    'Max_Creatinine': 165.96,
    'Min_Creatinine': 61.41,
    'First_Creatinine':77.08,
    'Last_Creatinine':165.96,
    'Creatinine_Mean': 89.53,
    'Creatinine_Std': 32.47604,
    'Creatinine_Change': 88.88,
    'Creatinine_CV': 0.36274,
    'Creatinine_Max_Change': 0.0,
    'Creatinine_Max_Consec_Increase': 72.01,
    'Creatinine_Max_Consec_Decrease':15.67,
    'Creatinine_Increase_Count': 5,
    'Creatinine_Count_Since_Peak': 0,
    'Creatinine_gt1p5_Count': 1,
    'Creatinine_Last_gt130_Count': 1,
    'To_Send_Positive_AKI_Msg_Timestamp': None
}]

DATABASE_LAB2_DATA_ALERT = [{
    'PID': 189386313, 
    'Age': 41,
    'Sex': 0,
    'Creatinine_Count': 8,
    'Max_Creatinine': 165.96,
    'Min_Creatinine': 61.41,
    'First_Creatinine':77.08,
    'Last_Creatinine':165.96,
    'Creatinine_Mean': 89.53,
    'Creatinine_Std': 32.47604,
    'Creatinine_Change': 88.88,
    'Creatinine_CV': 0.36274,
    'Creatinine_Max_Change': 0.0,
    'Creatinine_Max_Consec_Increase': 72.01,
    'Creatinine_Max_Consec_Decrease':15.67,
    'Creatinine_Increase_Count': 5,
    'Creatinine_Count_Since_Peak': 0,
    'Creatinine_gt1p5_Count': 1,
    'Creatinine_Last_gt130_Count': 1,
    'To_Send_Positive_AKI_Msg_Timestamp': '20240730125000'
}]


def compare_dataframes(df_actual, df_expected, tol=1e-6):
    """
    Helper Function to compare 2 dataframe, prints different rows
    """
    # Compare unequal data
    unequal_mask = pd.DataFrame(False, index=df_actual.index, columns=df_actual.columns)

    for col in df_actual.columns:
        if np.issubdtype(df_actual[col].dtype, np.number):
            unequal_mask[col] = np.abs(df_actual[col] - df_expected[col]) > tol
        else:
            unequal_mask[col] = df_actual[col] != df_expected[col]

    if unequal_mask.any().any():
        print("Differences found:")
        for row_idx in range(len(df_actual)):
            for col in df_actual.columns:
                if unequal_mask.at[row_idx, col]:
                    print(f"Row {row_idx}, Column '{col}': actual={df_actual.at[row_idx, col]}, expected={df_expected.at[row_idx, col]}")
        return False
    return True


class Integration_Test(unittest.TestCase):
    # Setup the connection 
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory and generate a messages.mllp file
        # Write the temporary messages into messages.mllp
        cls.directory = tempfile.mkdtemp()
        messages_filename = os.path.join(cls.directory, "messages.mllp")
        with open(messages_filename, "wb") as w:
            for m in (ADT_A01, ORU_R01, ORU_R02, ADT_A03):
                w.write(to_mllp(m))
        # Startup the simulator 
        cls.simulator = subprocess.Popen([
            "simulator/simulator.py",
            f"--mllp={TEST_MLLP_PORT}",
            f"--pager={TEST_PAGER_PORT}",
            f"--messages={messages_filename}"
        ])
        assert (wait_until_healthy(cls.simulator, f"localhost:{TEST_PAGER_PORT}"))
        # Initialise system
        cls.database_directory = tempfile.mkdtemp()
        cls.logs_directory = tempfile.mkdtemp()
        cls.system_engine = System_Engine(MLLP_HOST='localhost',
                                           MLLP_PORT=TEST_MLLP_PORT,
                                           PAGER_URL=f'http://localhost:{TEST_PAGER_PORT}/page',
                                           weights_filepath='./model/aki_model.pkl',
                                           db_filepath=cls.database_directory,
                                           history_filepath='./tests/data/mock_history.csv',
                                           system_logs_filepath=f'{cls.logs_directory}/system.log',
                                           message_logs_path=f'{cls.logs_directory}/message.log'
                                           )
        

    # Initialise system and check that history is processed correctly
    def test_system_initialisation(self):
        # Ensure database results meet expections
        select_query = f"""SELECT * FROM patients ORDER BY PID;"""
        database_data = pd.DataFrame.from_dict(self.system_engine.sqlite_manager.fetchall(sql=select_query))
        expected_data = pd.read_csv('./tests/data/mock_processed_history.csv')
        expected_data['To_Send_Positive_AKI_Msg_Timestamp'] = pd.NA
        # Ensure same column headers and reset index
        expected_data = expected_data[database_data.columns]        
        database_data = database_data.reset_index(drop=True)
        expected_data = expected_data.reset_index(drop=True)

        # Replace None with a mock value
        database_data = database_data.replace({None: False})
        expected_data = expected_data.where(pd.notnull(expected_data), False)
        result = compare_dataframes(database_data, expected_data)
        self.assertTrue(result)
    # -----------------------
    # NOTE: Only testing for the system workflows
    # Will not be testing for the other functionalities as unit test will ensure
    # -----------------------


    # Test for admit data
    def test_admission_workflow(self):
        message_list = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", TEST_MLLP_PORT))
            while True:
                buffer = s.recv(1024)
                if len(buffer) == 0:
                    break
                # Parse data
                parsed_msg, raw_msg = self.system_engine.message_parser.parse_mllp(buffer)
                message_list.append(raw_msg)
                # Only process the first message (admission) and simply acknowledge the rest
                if len(message_list) == 1:
                    start_time = time.perf_counter()
                    self.assertEqual(parsed_msg['MSH'][8], 'ADT^A01')
                    self.assertEqual(parsed_msg['PID'][3], '189386313')
                    # Ensure no error
                    self.assertFalse(self.system_engine._admit_process(parsed_msg['PID'][3], parsed_msg))
                    # Ensure the DB age and sex is updated accurately 
                    patient_data = self.system_engine.sqlite_manager.query({
                                                                        "table": "patients",
                                                                        "where": {"PID": 189386313}
                                                                    })
                    self.assertEqual(patient_data, DATABASE_ADMISSION_DATA)
                    end_time = time.perf_counter()

                # Acknowledge all messages
                s.sendall(self.system_engine.message_parser.format_response(False))

            if start_time is not None and end_time is not None:
                duration = end_time - start_time
                self.assertLess(duration, 3.0)


    # Test for negative laboratory data
    def test_negative_inference_workflow(self):
        message_list = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", TEST_MLLP_PORT))
            while True:
                buffer = s.recv(1024)
                if len(buffer) == 0:
                    break
                # Parse data
                parsed_msg, raw_msg = self.system_engine.message_parser.parse_mllp(buffer)
                message_list.append(raw_msg)
                # Only process the second message (negative lab) and simply acknowledge the rest
                if len(message_list) == 2:
                    # Start timing
                    start_time = time.perf_counter()
                    self.assertEqual(parsed_msg['MSH'][8], 'ORU^R01')
                    self.assertEqual(parsed_msg['PID'][3], '189386313')
                    # Ensure no error and alert (negative prediction)
                    error, alert = self.system_engine._inference_process(parsed_msg['PID'][3], parsed_msg)
                    self.assertFalse(error)
                    self.assertFalse(alert)
                    # Ensure the DB data accurately updated
                    patient_data = self.system_engine.sqlite_manager.query({
                                                                        "table": "patients",
                                                                        "where": {"PID": 189386313}
                                                                    })
                    self.assertEqual(patient_data, DATABASE_LAB1_DATA)
                    end_time = time.perf_counter()

                # Acknowledge all messages
                s.sendall(self.system_engine.message_parser.format_response(False))

            if start_time is not None and end_time is not None:
                duration = end_time - start_time
                self.assertLess(duration, 3.0)


    # Test for positive laboratory data
    def test_positive_inference_workflow(self):
        message_list = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", TEST_MLLP_PORT))
            while True:
                buffer = s.recv(1024)
                if len(buffer) == 0:
                    break
                # Parse data
                parsed_msg, raw_msg = self.system_engine.message_parser.parse_mllp(buffer)
                message_list.append(raw_msg)
                # Only process the third message (postive lab) and simply acknowledge the rest
                if len(message_list) == 3:
                    # Start timing
                    start_time = time.perf_counter()
                    self.assertEqual(parsed_msg['MSH'][8], 'ORU^R01')
                    self.assertEqual(parsed_msg['PID'][3], '189386313')
                    # Ensure no error and true alert (positive prediction)
                    error, alert = self.system_engine._inference_process(parsed_msg['PID'][3], parsed_msg)
                    self.assertFalse(error)
                    self.assertTrue(alert)
                    # Ensure the DB data accurately updated
                    patient_data = self.system_engine.sqlite_manager.query({
                                                                        "table": "patients",
                                                                        "where": {"PID": 189386313}
                                                                    })
                    self.assertEqual(patient_data, DATABASE_LAB2_DATA)
                    # Send alert and ensure successful append to database and queue
                    self.assertEqual(int(parsed_msg['MSH'][6]), 20240730125000)
                    self.system_engine._send_alert(str(parsed_msg['PID'][3]), str(parsed_msg['MSH'][6]))
                    # Check that queue updated
                    self.assertEqual(self.system_engine.alert_queue.qsize(), 1)
                    # Send alert and ensure successful (assert runs in _send_alert)
                    patient_data = self.system_engine.sqlite_manager.query({
                                                        "table": "patients",
                                                        "where": {"PID": 189386313}
                                                    })
                    self.assertEqual(patient_data, DATABASE_LAB2_DATA_ALERT)
                    
                    # Check for pager to send info
                    max_wait = 3
                    current_wait = 0
                    while not self.system_engine.alert_queue.empty() and current_wait < max_wait:
                        time.sleep(0.1)
                        current_wait += 0.1
                    
                    # Give it some time to just update, if the total latency <3s that is good
                    time.sleep(2)
                    # Check database updated again to remove the alert msg time
                    patient_data = self.system_engine.sqlite_manager.query({
                                                        "table": "patients",
                                                        "where": {"PID": 189386313}
                                                    })
                    self.assertEqual(patient_data, DATABASE_LAB2_DATA)
                    end_time = time.perf_counter()
                
                # Acknowledge all messages
                s.sendall(self.system_engine.message_parser.format_response(False))

            if start_time is not None and end_time is not None:
                duration = end_time - start_time
                self.assertLess(duration, 3.0)


    # Test for discharge workflow
    def test_discharge_workflow(self):
        message_list = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", TEST_MLLP_PORT))
            while True:
                buffer = s.recv(1024)
                if len(buffer) == 0:
                    break
                # Parse data
                parsed_msg, raw_msg = self.system_engine.message_parser.parse_mllp(buffer)
                message_list.append(raw_msg)
                # Only process the last message (discharge) and simply acknowledge the rest
                if len(message_list) == 4:
                    # Start timing
                    start_time = time.perf_counter()
                    self.assertEqual(parsed_msg['MSH'][8], 'ADT^A03')
                    self.assertEqual(parsed_msg['PID'][3], '189386313')
                    # Ensure no error and true alert (positive prediction)
                    self.assertFalse(self.system_engine._discharge_process(parsed_msg['PID'][3], parsed_msg))
                    # Ensure the DB data did not change
                    patient_data = self.system_engine.sqlite_manager.query({
                                                                        "table": "patients",
                                                                        "where": {"PID": 189386313}
                                                                    })
                    self.assertEqual(patient_data, DATABASE_LAB2_DATA)
                    end_time = time.perf_counter()

                # Acknowledge all messages
                s.sendall(self.system_engine.message_parser.format_response(False))

            if start_time is not None and end_time is not None:
                duration = end_time - start_time
                self.assertLess(duration, 3.0)


    # Test full system workflow
    def test_full_system(self):
        # Restart a new database and system
        shutil.rmtree(self.database_directory)
        aki_system = System_Engine(MLLP_HOST='localhost',
                                    MLLP_PORT=TEST_MLLP_PORT,
                                    PAGER_URL=f'http://localhost:{TEST_PAGER_PORT}/page',
                                    weights_filepath='./model/aki_model.pkl',
                                    db_filepath=self.database_directory,
                                    history_filepath='./tests/data/mock_history.csv',
                                    system_logs_filepath=f'{self.logs_directory}/system.log',
                                    message_logs_path=f'{self.logs_directory}/message.log'
                                    )

        # receive_message() now runs until stopped (it reconnects on a clean close rather
        # than returning), so drive it as a service: run it in a thread, wait until it has
        # made its positive prediction, then stop it and inspect the predictions returned.
        result = {}
        def run():
            result['preds'] = aki_system.receive_message()

        before = positive_predictions_total._value.get()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        deadline = time.time() + 30
        while time.time() < deadline and positive_predictions_total._value.get() <= before:
            time.sleep(0.2)

        aki_system.running = False
        worker.join(timeout=15)

        self.assertIn('preds', result)
        self.assertIn((189386313, 20240730125000), result['preds'])

    # Cleanup
    @classmethod
    def tearDownClass(cls):
        try:
            r = urllib.request.urlopen(f"http://localhost:{TEST_PAGER_PORT}/shutdown")
            assert (r.status==http.HTTPStatus.OK)
            cls.simulator.wait()
            assert (cls.simulator.returncode==0)

        finally:
            if cls.simulator.poll() is None:
                cls.simulator.kill()

            # Delete and cleanup
            if os.path.exists(cls.database_directory) and os.path.isdir(cls.database_directory):
                shutil.rmtree(cls.database_directory)

            if os.path.exists(cls.logs_directory) and os.path.isdir(cls.logs_directory):
                shutil.rmtree(cls.logs_directory)

            if os.path.exists(cls.directory) and os.path.isdir(cls.directory):
                shutil.rmtree(cls.directory)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Integration_Test('test_system_initialisation'))
    suite.addTest(Integration_Test('test_admission_workflow'))
    suite.addTest(Integration_Test('test_negative_inference_workflow'))
    suite.addTest(Integration_Test('test_positive_inference_workflow'))
    suite.addTest(Integration_Test('test_discharge_workflow'))
    suite.addTest(Integration_Test('test_full_system'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
