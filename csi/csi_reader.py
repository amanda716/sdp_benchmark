
import numpy as np
from csi.csi_record import CSI_Config


class CSI_Reader:
    """Class to read and parse CSI records from a text file."""
    BUFFER_SIZE = 1024 * 1024  # 1MB buffer size

    def __init__(self, config: CSI_Config):
        self.config = config
        self.records = []

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
