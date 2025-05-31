from sdp.reader.readers.bfee_reader import BfeeReader
from sdp.reader.readers.hw_office_reader import HwOfficeReader
from sdp.reader.readers.wiprox_mat_reader import WiproxMatReader


class ReaderFactory:
    READERS = [BfeeReader,
               WiproxMatReader,
               HwOfficeReader]

    @classmethod
    def create_reader(cls, file_path: str):
        for reader_cls in cls.READERS:
            if reader_cls.can_read(file_path):
                return reader_cls(file_path)
        raise ValueError(f"Unsupported file format: {file_path}")

    # def get_reader(path: str) -> Reader:
    #     for reader in READERS:
    #         if reader.can_read(path):
    #             return reader()
    #
    #     print("Unable to automatically select a reader.")
    #     print("Defaulting to Intel format.")
    #
    #     return None
