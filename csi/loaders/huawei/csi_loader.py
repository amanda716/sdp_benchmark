
import os
import numpy as np
from csi.csi_domain import CsiConfig


class CsiDataLoader:
    """Class to read and parse CSI records from a text file."""
    BUFFER_SIZE = 1024 * 1024  # 1MB buffer size

    def __init__(self, config: CsiConfig):
        self.config = config
        self.records = []

    def read_records_from_folder(self, folder: str):
        """
        Reads and processes records from all text files in a specified folder, 
        excluding files that contain '_truth' in their name.
        Args:
            folder (str): The path to the folder containing the text files.
        Returns:
            list: A list of dictionaries, where each dictionary contains:
                - "file_path" (str): The path of the processed file.
                - "records" (list): A list of records read from the file.
        """

        file_data = []
        for filename in os.listdir(folder):
            if filename.endswith(".txt") and '_truth' not in filename:
                # Skip files that contain '_truth' in their name
                # and only process text files
                file_path = os.path.join(folder, filename)
                file_data.append({
                    "file_path": file_path,
                    "records": list(self.read_records(file_path))
                })
        return file_data

    def random_check(self, file_data):
        for data in file_data:
            filename = data["filename"]
            records = data["records"]
            if len(records) > 0:
                # Randomly check a record
                record = records[0]
                print(f"Filename: {filename}, First Record: {record}")
            else:
                print(f"No records found in {filename}")

    def read_records(self, filename: str):
        """
        Reads and parses records from a text file.

        This method reads a file in chunks, processes the content to extract
        records of a specified length, and yields parsed records one by one.
        Any incomplete record at the end of the file will trigger a warning.

        Args:
            filename (str): The path to the text file to be read.

        Yields:
            Parsed records as determined by the `_parse_internal` method.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If there is an error reading the file.

        Notes:
            - The `record_length` is determined by the `self.config.record_length`.
            - The `BUFFER_SIZE` determines the size of chunks read from the file.
            - Any remaining words that do not form a complete record will not be
              yielded but will instead trigger a warning message.
        """
        record_length = self.config.record_length
        with open(filename, 'r', encoding="utf-8") as f:
            word_buffer = ""
            while True:
                chunk = f.read(self.BUFFER_SIZE)
                if not chunk:  # End of file
                    break
                word_buffer += chunk
                words = word_buffer.split()
                while len(words) >= record_length:
                    yield self._parse_internal(words[:record_length])
                    words = words[record_length:]

                word_buffer = " ".join(words)
            if word_buffer:
                # yield word_buffer.split()
                # Handle any remaining words in the buffer
                print(
                    f"Warning: Incomplete record at the end of file. Remaining words: {len(words)}")

    def _parse_internal(self, chunk: list) -> dict:
        """Parse a single record and return a dictionary of values."""

        # 逐字段解析
        # 1) ts
        ts_count = self.config.ts_count
        ts = [float(x) for x in chunk[0:ts_count]]

        # 2) rssi
        rssi_count = self.config.rssi_count
        rssi_start = ts_count
        rssi = [float(x) for x in chunk[rssi_start: rssi_start + rssi_count]]

        # 3) mcs
        mcs_count = self.config.mcs_count
        mcs_start = ts_count + rssi_count
        mcs_values = chunk[mcs_start: mcs_start + mcs_count]
        mcs = float(mcs_values[0]) if len(mcs_values) > 0 else 0

        # 4) gain
        gain_count = self.config.gain_count
        gain_start = mcs_start + mcs_count
        gain = [float(x) for x in chunk[gain_start: gain_start+gain_count]]

        # 5) csi
        csi_count = self.config.csi_count
        csi_start = gain_start + gain_count
        csi = list(map(self._parse_or_zero,
                   chunk[csi_start: csi_start + csi_count]))
        csi = np.array(csi, dtype=complex)
        n_subcarriers = self.config.csi_num_subcarriers
        n_antennas = self.config.csi_num_antennas
        csi = csi.reshape((n_subcarriers, n_antennas))

        # 6) frequency
        final_ts = self.config.target_fs

        return {
            "ts": ts,
            "rssi": rssi,
            "mcs": mcs,
            "gain": gain,
            "csi": csi,
            "frequency": final_ts
        }

    def _parse_or_zero(self, s):
        try:
            # Replace 'I' or 'i' with 'j' for complex number notation
            s_replaced = s.replace('I', 'j').replace('i', 'j')
            return complex(s_replaced)
        except ValueError:
            # Return 0+0j if parsing fails
            return 0+0j
        

class CsiTruthDataLoader:
    """Class to read and parse truth files."""

    @staticmethod
    def read_truth_file(truth_path: str):
        """
        Reads and parses a truth file.

        Args:
            filepath (str): The path to the truth file.

        Returns:
            list: A list of dictionaries, where each dictionary represents a truth record.
        """
        data = []
        with open(truth_path, 'r', encoding="utf-8") as f:
           data = f.read().strip().split()
        return [int(x) for x in data]
