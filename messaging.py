# Relevant imports
from datetime import datetime

MLLP_START_OF_BLOCK = 0x0b
MLLP_END_OF_BLOCK = 0x1c
MLLP_CARRIAGE_RETURN = 0x0d

class Message_Parser:
    def __init__(self):
        """
        Class that handles all parsing of HL7 and formatting of acknowledgment message
        """

    def parse_mllp(self, buffer):
        """
        Function to parse mllp buffer message
        - Converts mllp into a dictionary in the format of

        Eg. MSH|^~\\&|SIMULATION|SOUTH RIVERSIDE|||20240211075100||ADT^A01|||2.5
            PID|1||170910912||ALAIA WATKINS||19910319|F

        To: 
        {
            MSH: [MSH, ^~\\&, SIMULATION, SOUTH RIVERSIDE, , , 20240211075100, , ADT^A01, ...],
            PID: [PID, 1, 170910912, , ALAIA WATKINS, , 19910319, F]
        }

        Returns:
            parsed_message (dict)
            raw_message (str)
        """
        hl7 = str(buffer[1:-3], "ascii").split("\r") # Strip MLLP framing and final \r
        # Convert into aa hl7 dictionary for easy access to data
        # Eg. {MLP: {}, PID:{}, ...}
        hl7_dict = {}
        for segment in hl7:
            fields = segment.split("|")
            key = fields[0]
            hl7_dict[key] = fields
        return hl7_dict, hl7

    def extract_frames(self, buffer):
        """
        Split a byte buffer into complete MLLP frames.

        MLLP frames are START_OF_BLOCK (0x0b) ... END_OF_BLOCK (0x1c) CARRIAGE_RETURN (0x0d).
        A single TCP read may contain only part of a frame (e.g. when the sender splits a
        message) or several frames. Returns (frames, remainder): a list of complete frames -
        each still including its framing bytes, ready for parse_mllp - and the trailing
        incomplete bytes to carry over to the next read.
        """
        frames = []
        end = bytes([MLLP_END_OF_BLOCK, MLLP_CARRIAGE_RETURN])
        while True:
            idx = buffer.find(end)
            if idx == -1:
                break
            frame_end = idx + len(end)
            frame, buffer = buffer[:frame_end], buffer[frame_end:]
            # Drop any bytes before the start-of-block for robustness.
            start = frame.find(MLLP_START_OF_BLOCK)
            if start != -1:
                frames.append(frame[start:])
        return frames, buffer

    def format_response(self, error:bool):
        """
        Function to format response to pager
        - Sends the PID,datetime of message
        - Sends AA if no error, else sends AE

        Args:
            error (bool): Flag if error occured during the model workflow

        Returns:
            m (bytes): Formatted message to send to pager
        """
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        ACK = [
            f"MSH|^~\\&|||||{current_time}||ACK|||2.5",
            f"MSA|{'AE' if error else 'AA'}",
        ]
        m = bytes(chr(MLLP_START_OF_BLOCK), "ascii")
        m += bytes("\r".join(ACK) + "\r", "ascii")
        m += bytes(chr(MLLP_END_OF_BLOCK) + chr(MLLP_CARRIAGE_RETURN), "ascii")
        return m