# Failure-mode tests (roadmap Phase 0 + Phase 1).
#
# These exercise the reliability paths the CW5 spec requires us to handle:
#   - pager request failing then succeeding              (#3)
#   - unsent alerts surviving a restart                  (#7)
#   - MLLP peer closing the connection -> reconnect      (#1)
#   - a malformed message -> AE NAK, loop survives       (#5, #6)
#
# They use two controllable test doubles instead of the real simulator/pager so we
# can inject faults deterministically: a StubPager (HTTP) and a FakeMllpServer (TCP).

import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from database import SQLite_Manager
from main import System_Engine
from simulator.simulator_test import ADT_A03, to_mllp


class StubPager:
    """A controllable pager endpoint.

    Records every POSTed body. The first `fail_times` requests return HTTP 500 so we
    can test retry/backoff; subsequent requests return 200 and are recorded. Binds an
    ephemeral port so tests never collide.
    """

    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = 0
        self.received = []
        self._server = HTTPServer(('localhost', 0), self._make_handler())
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self):
        return f"http://localhost:{self.port}/page"

    def _make_handler(self):
        pager = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length).decode('utf-8')
                pager.calls += 1
                if pager.calls <= pager.fail_times:
                    self.send_response(500)
                    self.end_headers()
                    return
                pager.received.append(body)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass  # keep test output quiet

        return Handler

    def stop(self):
        self._server.shutdown()


class FakeMllpServer:
    """A scriptable MLLP server.

    `connection_scripts` is a list of scripts, one per accepted connection; each script
    is a list of framed MLLP message bytes to send, reading one ACK between each. After
    every script's messages are sent the connection is closed (so the client sees a
    zero-length read and must reconnect). Once all scripts are exhausted it keeps
    accepting and immediately closing connections, so a stopped engine can exit between
    reconnect attempts. Collected ACK buffers are exposed on `acks`.
    """

    def __init__(self, connection_scripts, split=False):
        self.connection_scripts = list(connection_scripts)
        self.split = split  # if True, send each frame in two halves (like --short_messages)
        self.acks = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('localhost', 0))
        self._sock.listen(5)
        self._sock.settimeout(0.5)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _send(self, conn, msg):
        if self.split and len(msg) > 1:
            mid = len(msg) // 2
            conn.sendall(msg[:mid])
            time.sleep(0.3)
            conn.sendall(msg[mid:])
        else:
            conn.sendall(msg)

    def _accept(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
                return conn
            except socket.timeout:
                continue
            except OSError:
                return None
        return None

    def _serve(self):
        for script in self.connection_scripts:
            conn = self._accept()
            if conn is None:
                return
            try:
                for msg in script:
                    self._send(conn, msg)
                    ack = conn.recv(1024)
                    if ack:
                        self.acks.append(ack)
            except OSError:
                pass
            finally:
                conn.close()
        # Drain: accept and immediately close so the client keeps reconnecting.
        while not self._stop.is_set():
            conn = self._accept()
            if conn is None:
                return
            conn.close()

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


class FailureModeTests(unittest.TestCase):
    def _make_engine(self, pager_url, mllp_port=1, db_dir=None):
        db_dir = db_dir or tempfile.mkdtemp()
        logs = tempfile.mkdtemp()
        engine = System_Engine(
            MLLP_HOST='localhost',
            MLLP_PORT=mllp_port,
            PAGER_URL=pager_url,
            weights_filepath='./model/aki_model.pkl',
            db_filepath=db_dir,
            history_filepath='./tests/data/mock_history.csv',
            system_logs_filepath=f'{logs}/system.log',
            message_logs_path=f'{logs}/message.log',
        )
        self.addCleanup(lambda: setattr(engine, 'running', False))
        return engine, db_dir

    def _wait_for(self, predicate, timeout=15, interval=0.1):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return False

    def _any_pid(self, engine):
        return engine.sqlite_manager.fetchone("SELECT PID FROM patients LIMIT 1")['PID']

    # --- #3: pager fails then succeeds ---
    def test_pager_retries_until_success(self):
        pager = StubPager(fail_times=2)
        self.addCleanup(pager.stop)
        engine, _ = self._make_engine(pager.url)
        pid = self._any_pid(engine)

        engine._send_alert(str(pid), "20240730125000")

        self.assertTrue(self._wait_for(lambda: pager.received, timeout=25),
                        "alert was never delivered after retries")
        self.assertEqual(pager.received, [f"{pid},20240730125000"])
        self.assertGreaterEqual(pager.calls, 3)  # 2 failures + 1 success

        # The persisted alert must be cleared once delivered.
        def column_cleared():
            row = engine.sqlite_manager.fetchone(
                "SELECT To_Send_Positive_AKI_Msg_Timestamp AS t FROM patients WHERE PID = ?", (pid,))
            return row['t'] is None
        self.assertTrue(self._wait_for(column_cleared), "DB alert column not cleared")

    # --- #7: unsent alerts replayed on restart ---
    def test_unsent_alerts_replayed_on_startup(self):
        pager = StubPager()
        self.addCleanup(pager.stop)
        db_dir = tempfile.mkdtemp()

        # First engine creates the DB; we persist an unsent alert directly, then stop it.
        engine1, _ = self._make_engine(pager.url, db_dir=db_dir)
        engine1.running = False
        pid = self._any_pid(engine1)
        engine1.sqlite_manager.update({
            "table": "patients",
            "set": {"To_Send_Positive_AKI_Msg_Timestamp": "20240730125000"},
            "where": {"PID": pid},
        })
        engine1.shutdown()

        # A fresh engine on the same DB must replay the persisted alert on startup.
        self._make_engine(pager.url, db_dir=db_dir)

        self.assertTrue(self._wait_for(lambda: pager.received),
                        "persisted alert was not replayed on startup")
        self.assertEqual(pager.received, [f"{pid},20240730125000"])

    # --- #1: peer closes the connection -> engine reconnects ---
    def test_reconnects_after_peer_closes(self):
        pager = StubPager()
        self.addCleanup(pager.stop)
        good = to_mllp(ADT_A03)  # discharge: always ACK AA, no alert, no patient needed
        server = FakeMllpServer([[good], [good]])  # two separate connections
        self.addCleanup(server.stop)
        engine, _ = self._make_engine(pager.url, mllp_port=server.port)

        t = threading.Thread(target=engine.receive_message, daemon=True)
        t.start()
        try:
            self.assertTrue(self._wait_for(lambda: len(server.acks) >= 2, timeout=20),
                            "engine did not reconnect after the peer closed the connection")
            for ack in server.acks[:2]:
                self.assertIn(b'MSA|AA', ack)
        finally:
            engine.running = False
            server.stop()
            t.join(timeout=10)

    # --- #5/#6: malformed message -> AE NAK, loop survives and processes the next one ---
    def test_malformed_message_is_naked_and_loop_survives(self):
        pager = StubPager()
        self.addCleanup(pager.stop)
        malformed = to_mllp(["MSH|^~\\&|SIMULATION|SOUTH RIVERSIDE|||20240101000000||ORU^R01|||2.5"])  # no PID/OBX
        good = to_mllp(ADT_A03)
        server = FakeMllpServer([[malformed, good]])  # both on one connection
        self.addCleanup(server.stop)
        engine, _ = self._make_engine(pager.url, mllp_port=server.port)

        t = threading.Thread(target=engine.receive_message, daemon=True)
        t.start()
        try:
            self.assertTrue(self._wait_for(lambda: len(server.acks) >= 2, timeout=20),
                            "engine did not stay alive after a malformed message")
            self.assertIn(b'MSA|AE', server.acks[0])  # malformed -> NAK
            self.assertIn(b'MSA|AA', server.acks[1])  # next message still processed
        finally:
            engine.running = False
            server.stop()
            t.join(timeout=10)

    # --- #13: a message split across TCP reads is reassembled and ACKed once ---
    def test_split_message_is_reassembled(self):
        pager = StubPager()
        self.addCleanup(pager.stop)
        good = to_mllp(ADT_A03)
        server = FakeMllpServer([[good]], split=True)
        self.addCleanup(server.stop)
        engine, _ = self._make_engine(pager.url, mllp_port=server.port)

        t = threading.Thread(target=engine.receive_message, daemon=True)
        t.start()
        try:
            self.assertTrue(self._wait_for(lambda: len(server.acks) >= 1, timeout=20),
                            "engine did not reassemble a split MLLP message")
            self.assertIn(b'MSA|AA', server.acks[0])
        finally:
            engine.running = False
            server.stop()
            t.join(timeout=10)

    # --- #4: SIGTERM/shutdown is bounded and never drops an undelivered alert ---
    def test_undelivered_alert_persists_through_shutdown(self):
        pager = StubPager(fail_times=10_000)  # never succeeds
        self.addCleanup(pager.stop)
        db_dir = tempfile.mkdtemp()
        engine, _ = self._make_engine(pager.url, db_dir=db_dir)
        pid = self._any_pid(engine)

        engine._send_alert(str(pid), "20240730125000")
        time.sleep(1.0)  # let the worker attempt (and fail) at least once

        start = time.time()
        engine.shutdown()
        self.assertLess(time.time() - start, 16, "shutdown did not complete promptly")
        self.assertFalse(engine.alert_worker.is_alive())

        # The undelivered alert must still be persisted (so it replays on next start).
        manager = SQLite_Manager(db_dir)
        manager._initialise_database()
        self.addCleanup(manager.close)
        row = manager.fetchone(
            "SELECT To_Send_Positive_AKI_Msg_Timestamp AS t FROM patients WHERE PID = ?", (pid,))
        self.assertEqual(row['t'], "20240730125000")

    # --- #9: concurrent _send_alert calls do not lose updates to shared state ---
    def test_concurrent_send_alert_no_lost_updates(self):
        pager = StubPager()
        self.addCleanup(pager.stop)
        engine, _ = self._make_engine(pager.url)
        pid = self._any_pid(engine)

        n = 15
        timestamps = [f"2024073012{i:04d}" for i in range(n)]
        threads = [threading.Thread(target=engine._send_alert, args=(str(pid), ts))
                   for ts in timestamps]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertTrue(self._wait_for(lambda: len(pager.received) >= n, timeout=30),
                        "not all concurrently-queued alerts were delivered")
        self.assertEqual(sorted(pager.received), sorted(f"{pid},{ts}" for ts in timestamps))


if __name__ == "__main__":
    unittest.main()
