
from csi.csi_record import CSI_Config


class CSI_Reader:
    """Class to read and parse CSI records from a binary file."""

    def __init__(self, filename: str, config: CSI_Config):
        self.filename = filename
        self.config = config
        self.records = []

    def read_records(self):
        """Read and parse the records from the binary file."""
        with open(self.filename, 'r') as f:
            while True:
                record = f.read(self.config.record_length)
                if not record:
                    break
                self.records.append(record)