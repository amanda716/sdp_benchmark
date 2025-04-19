import json


class CSI_Config:
    """Class representing the structure of a CSI record."""

    def __init__(self,
                 record_length=268,
                 ts_count=3,
                 rssi_count=4,
                 mcs_count=1,
                 gain_count=4,
                 csi_count=64 * 4,
                 csi_num_subcarriers=64,
                 csi_num_antennas=4,
                 target_fs=100):
        """Initialize the Record with configurable parameters."""
        self.record_length = record_length
        self.ts_count = ts_count
        self.rssi_count = rssi_count
        self.mcs_count = mcs_count
        self.gain_count = gain_count
        self.csi_count = csi_count
        self.csi_num_subcarriers = csi_num_subcarriers
        self.csi_num_antennas = csi_num_antennas
        self.target_fs = target_fs

    @classmethod
    def create_config(cls, config):
        return cls(**config)

    def to_dict(self):
        """
        Converts the attributes of the object into a dictionary representation.
        Returns:
            dict: A dictionary containing the following key-value pairs:
                - "record_length" (int): The length of the record.
                - "ts_count" (int): The timestamp count.
                - "rssi_count" (int): The RSSI (Received Signal Strength Indicator) count.
                - "mcs_count" (int): The MCS (Modulation and Coding Scheme) count.
                - "gain_count" (int): The gain count.
                - "csi_count" (int): The CSI (Channel State Information) count.
                - "csi_num_subcarriers" (int): The number of subcarriers in the CSI.
                - "csi_num_antennas" (int): The number of antennas in the CSI.
                - "target_fs" (int): The target sampling frequency.
        """
        
        return {
            "record_length": self.record_length,
            "ts_count": self.ts_count,
            "rssi_count": self.rssi_count,
            "mcs_count": self.mcs_count,
            "gain_count": self.gain_count,
            "csi_count": self.csi_count,
            "csi_num_subcarriers": self.csi_num_subcarriers,
            "csi_num_antennas": self.csi_num_antennas,
            "target_fs": self.target_fs
        }

    def save_to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load_config_from_file(cls, filename):
        """Load configuration from a JSON file and create a new instance."""
        try:
            with open(filename, 'r') as file:
                config = json.load(file)
            return cls(**config)
        except FileNotFoundError:
            print(f"Error: Configuration file '{filename}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{filename}'.")
            return None
