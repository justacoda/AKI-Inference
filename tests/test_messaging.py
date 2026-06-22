import unittest
from datetime import datetime
from messaging import Message_Parser
from simulator.simulator_test import (ADT_A01, ORU_R01,ADT_A03, to_mllp)

# Message templates from simulator_test
ADT_A01_RAW = [
    "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||202401201630||ADT^A01|||2.5",
    "PID|1||478237423||ELIZABETH HOLMES||19840203|F",
    "NK1|1|SUNNY BALWANI|PARTNER"
]

ORU_R01_RAW = [
    "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||202401201800||ORU^R01|||2.5",
    "PID|1||478237423",
    "OBR|1||||||202401202243",
    "OBX|1|SN|CREATININE||103.4",
]

ADT_A03_RAW = [
    "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||202401221000||ADT^A03|||2.5",
    "PID|1||478237423",
]

message_parser = Message_Parser()


class Messaging_Test(unittest.TestCase):
    # Test that key parsing information for ADT_A01 is correct
    def test_admit_message_data(self):
        received_msg = to_mllp(ADT_A01)
        parsed_msg, raw_msg = message_parser.parse_mllp(received_msg)
        # Test Raw_msg
        self.assertEqual(raw_msg, ADT_A01_RAW)

        # Test Message Type
        self.assertEqual(parsed_msg['MSH'][8], 'ADT^A01')

        # Test Message Timestamp
        self.assertEqual(parsed_msg['MSH'][6], '202401201630')

        # Test PID 
        self.assertEqual(parsed_msg['PID'][3], '478237423')

        # Test DOB
        self.assertEqual(parsed_msg['PID'][7], '19840203')

        # Test Gender
        self.assertEqual(parsed_msg['PID'][8], 'F')

    
    # Test that key parsing information for ADT_A03 is correct
    def test_discharge_message_data(self):
        received_msg = to_mllp(ADT_A03)
        parsed_msg, raw_msg = message_parser.parse_mllp(received_msg)
        # Test Raw_msg
        self.assertEqual(raw_msg, ADT_A03_RAW)

        # Test Message Type
        self.assertEqual(parsed_msg['MSH'][8], 'ADT^A03')

        # Test Message Timestamp
        self.assertEqual(parsed_msg['MSH'][6], '202401221000')

        # Test PID 
        self.assertEqual(parsed_msg['PID'][3], '478237423')


    # Test that key parsing information for ORU_R01 is correct
    def test_labortary_message_data(self):
        received_msg = to_mllp(ORU_R01)
        parsed_msg, raw_msg = message_parser.parse_mllp(received_msg)
        # Test Raw_msg
        self.assertEqual(raw_msg, ORU_R01_RAW)

        # Test Message Type
        self.assertEqual(parsed_msg['MSH'][8], 'ORU^R01')

        # Test Message Timestamp
        self.assertEqual(parsed_msg['MSH'][6], '202401201800')

        # Test PID 
        self.assertEqual(parsed_msg['PID'][3], '478237423')

        # Test OBR 
        self.assertEqual(parsed_msg['OBR'][7], '202401202243')

        # Test OBX 
        self.assertEqual(parsed_msg['OBX'][5], '103.4')

    # Test acknowledgement format for no error
    def test_acknowledgement_format_no_error(self):
        ack_format = message_parser.format_response(error=False)
        parsed_ack, raw_ack = message_parser.parse_mllp(ack_format)
        self.assertEqual(raw_ack[0], f"MSH|^~\&|||||{datetime.now().strftime('%Y%m%d%H%M%S')}||ACK|||2.5")
        self.assertEqual(raw_ack[1], "MSA|AA")
        self.assertEqual(parsed_ack['MSA'][1], "AA")

    # Test acknowledgement format with error
    def test_acknowledgement_format_with_error(self):
        ack_format = message_parser.format_response(error=True)
        parsed_ack, raw_ack = message_parser.parse_mllp(ack_format)
        self.assertEqual(raw_ack[0], f"MSH|^~\&|||||{datetime.now().strftime('%Y%m%d%H%M%S')}||ACK|||2.5")
        self.assertEqual(raw_ack[1], "MSA|AE")
        self.assertEqual(parsed_ack['MSA'][1], "AE")


if __name__ == "__main__":
    unittest.main()
