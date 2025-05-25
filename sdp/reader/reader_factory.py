
from sdp.reader.reader import Reader
from sdp.reader.readers.bfee_reader import BfeeReader
from sdp.reader.readers.hw_office_reader import HwOfficeReader
from sdp.reader.readers.wiprox_mat_reader import WiproxMatReader


READERS = [BfeeReader,
           WiproxMatReader,
           HwOfficeReader]

def get_reader(path: str) -> Reader:
    for reader in READERS:
        if reader.can_read(path):
            return reader()

    print("Unable to automatically select a reader.")
    print("Defaulting to Intel format.")

    return None