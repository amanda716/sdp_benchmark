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

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    @classmethod
    def can_read(cls, file_path: str) -> bool:
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
        """
        参考 Intel 5300 read_bfee.c/read_bfee_new.c 的逻辑，对单条 BFEE payload 做解析。
        返回 bfee_dict: {
          'timestamp_low': int,
          'Nrx': int, 'Ntx': int,
          'rssi_a': int, 'rssi_b': int, 'rssi_c': int,
          'noise': int(有符号),
          'csi': shape=(30, Nrx, Ntx), dtype=complex64,
          ...
        }
        """
        file_name = os.path.basename(file_path)
        ret_data = CSIData(file_name)

        with open(file_path, 'rb') as f:
            filesize = os.fstat(f.fileno()).st_size
            cur = 0
            while (cur + 3) < filesize:
                hdr = f.read(3)
                if len(hdr) < 3: break
                field_len = (hdr[0] << 8) | hdr[1]
                code = hdr[2]
                cur += 3
                if code == 0xBB:
                    payload = f.read(field_len - 1)
                    cur += (field_len - 1)
                    if len(payload) < (field_len - 1):
                        break
                    frame = self.parse_bfee_record(payload)
                    if frame is not None:
                        ret_data.add_frame(frame)
                else:
                    f.seek(field_len - 1, 1)
                    cur += (field_len - 1)
        print(f"[Info] {file_name}: B_FEE records={len(ret_data.frames)}")
        return ret_data

    @staticmethod
    def parse_bfee_record(payload: bytes):
        """
        参考 Intel 5300 read_bfee.c/read_bfee_new.c 的逻辑，对单条 BFEE payload 做解析。
        返回 bfee_dict: {
          'timestamp_low': int,
          'Nrx': int, 'Ntx': int,
          'rssi_a': int, 'rssi_b': int, 'rssi_c': int,
          'noise': int(有符号),
          'csi': shape=(30, Nrx, Ntx), dtype=complex64,
          ...
        }
        """
        if len(payload) < 20:
            return None

        timestamp_low = (payload[0] |
                         (payload[1] << 8) |
                         (payload[2] << 16) |
                         (payload[3] << 24)) & 0xffffffff
        bfee_count = (payload[4] | (payload[5] << 8)) & 0xffff

        Nrx = payload[8]
        Ntx = payload[9]
        rssi_a = payload[10]
        rssi_b = payload[11]
        rssi_c = payload[12]
        noise = struct.unpack('b', payload[13:14])[0]
        agc = payload[14]
        antenna_sel = payload[15]
        csi_len = (payload[16] | (payload[17] << 8)) & 0xffff
        fake_rate = (payload[18] | (payload[19] << 8)) & 0xffff

        calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) // 8
        if csi_len != calc_len: return None
        if len(payload) < (20 + csi_len): return None

        csi_bytes = payload[20: 20 + csi_len]
        csi_array = np.zeros((30, Nrx, Ntx), dtype=np.complex64)

        bit_index = 0

        def get_bit(pos):
            byte_i = pos // 8
            if byte_i >= len(csi_bytes):
                return 0
            shift = pos % 8
            return (csi_bytes[byte_i] >> shift) & 0x1

        def get_bits_u8(pos):
            val = 0
            for b in range(8):
                val |= (get_bit(pos + b) << b)
            return val

        for sc_idx in range(30):
            bit_index += 3  # skip pilot
            for j in range(Nrx * Ntx):
                real8 = get_bits_u8(bit_index)
                imag8 = get_bits_u8(bit_index + 8)
                bit_index += 16
                if real8 & 0x80: real8 -= 256
                if imag8 & 0x80: imag8 -= 256
                rx_i = j % Nrx
                tx_i = j // Nrx
                csi_array[sc_idx, rx_i, tx_i] = np.complex64(real8 + 1j * imag8)
        return BfeeFrame(timestamp_low, bfee_count, Nrx, Ntx, rssi_a, rssi_b, rssi_c,
                         noise, csi_array, agc, antenna_sel, fake_rate)
