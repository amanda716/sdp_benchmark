import os
import struct

from bitstring import BitArray
import numpy as np
from csi.csi_data import CSIData
from csi.frames.bfee_frame import BfeeFrame
from sdp.reader.reader import Reader


SIZE_STRUCT = struct.Struct(">H").unpack
CODE_STRUCT = struct.Struct("B").unpack

HEADER_STRUCT = struct.Struct("<LHHBBBBBbBBHH").unpack
VALID_BEAMFORMING_MEASUREMENT = 0xBB


class BfeeReader(Reader):
    """
    Reader for WiDAR bfee files.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def can_read(self, file_path: str) -> bool:
        """
        Check if the reader can read the file at the given path.

        Args:
            path (str): The path to the file to check.

        Returns:
            bool: True if the reader can read the file, False otherwise.
            :param self:
            :param file_path: The path to the file to check
        """
        with open(file_path, 'rb') as f:
            data = f.read(3)
        if len(data) < 3:
            return False
        size = SIZE_STRUCT(data[:2])[0]
        code = CODE_STRUCT(data[2:3])[0]
        if size < 20 or code != VALID_BEAMFORMING_MEASUREMENT:
            return False
        return file_path.endswith('.dat')

    def read_file(self, file_path: str) -> CSIData:
        file_name = os.path.basename(file_path)
        ret_data = CSIData(file_name)

        with open(file_path, 'rb') as f:
            data = f.read()

        length = len(data)
        cursor = 0
        while length - cursor > 100:
            size = SIZE_STRUCT(data[cursor:cursor + 2])[0]
            code = CODE_STRUCT(data[cursor + 2:cursor + 3])[0]
            cursor += 3

            if code == VALID_BEAMFORMING_MEASUREMENT:
                all_blocks = data[cursor:cursor + size - 1]
                header_block = HEADER_STRUCT(all_blocks[:20])
                data_block = all_blocks[20:]
                n_tx = header_block[3]
                n_rx = header_block[4]
                expected_length = header_block[11]
                csi_matrix = self.parse_as_bfee_frame(
                    data_block, n_rx, n_tx, expected_length)
                if csi_matrix is not None:
                    frame = BfeeFrame(header_block, csi_matrix)
                    ret_data.add_frame(frame=frame)
            cursor += size - 1
        return ret_data

    @staticmethod
    def parse_as_bfee_frame(payload: bytes, n_rx: int, n_tx: int, expected_length: int) -> np.array:
        header_length = 20
        n_subcarriers = 30
        bits_per_component = 8
        n_components = 2
        pilot_bits = 3
        n_rx_tx_pairs = n_rx * n_tx
        calculated_byte_length = (n_subcarriers * n_rx_tx_pairs *
                                  n_bits_per_component * n_components + pilot_bits) + 7 // 8  # type: ignore

        if expected_length != calculated_byte_length:
            return None
        if len(payload) != expected_length + header_length:
            return None

        csi_bytes = payload[header_length: header_length+expected_length]
        csi_matrix = np.zeros((n_subcarriers, n_rx,
                               n_tx), dtype=np.complex64)

        csi_bitstream = BitArray(bytes=csi_bytes)

        bit_index = 0
        for sc_index in range(n_subcarriers):
            bit_index += pilot_bits
            for j in range(n_rx_tx_pairs):
                real8 = csi_bitstream[bit_index:bit_index +
                                      bits_per_component].uint
                imag8 = csi_bitstream[bit_index +
                                      bits_per_component:bit_index + 2 * bits_per_component].uint
                bit_index += 2 * bits_per_component
                if real8 > 127:
                    real8 -= 256
                if imag8 > 127:
                    imag8 -= 256
                rx_i = j % n_rx
                tx_i = j // n_tx
                csi_matrix[sc_index, rx_i, tx_i] = np.complex64(real8, imag8)

        return csi_matrix
