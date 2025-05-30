import os

import numpy as np
from csi.csi_data import CSIData
from csi.frames.hw_office_frame import HwOfficeFrame
from sdp.reader.reader import Reader

FILENAME_STRUCTURE_MAPPING = {
    268: {
        'record_length': 268,
        'ts_count': 3,  # hour, minute, second
        'rssi_count': 4,  # 4个RSSI
        'mcs_count': 1,
        'gain_count': 4,
        'csi_count': 64 * 4,  # 256
        'csi_num_subcarriers': 64,
        'csi_num_antennas': 4,
        'target_fs': 100  # 目标采样率 100
    },
    1000: {
        'record_length': 1000,
        'ts_count': 3,
        'rssi_count': 2,  # 2个RSSI
        'mcs_count': 1,
        'gain_count': 2,
        'csi_count': 248 * 4,  # 992
        'csi_num_subcarriers': 248,
        'csi_num_antennas': 4,
        'target_fs': 20  # 目标采样率 20
    },
    1008: {
        'record_length': 1008,
        'ts_count': 3,
        'rssi_count': 2,
        'mcs_count': 1,
        'gain_count': 2,
        'csi_count': 250 * 4,  # 1000
        'csi_num_subcarriers': 250,
        'csi_num_antennas': 4,
        'target_fs': 20
    }
}


def _parse_or_zero(s):
    try:
        # Replace 'I' or 'i' with 'j' for complex number notation
        s_replaced = s.replace('I', 'j').replace('i', 'j')
        return complex(s_replaced)
    except ValueError:
        # Return 0+0j if parsing fails
        return 0 + 0j


class HwOfficeReader(Reader):
    """
    Reader for HW Office documents.
    """

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        # Extract the record length from the filename
        self.file_structure = None
        for key in FILENAME_STRUCTURE_MAPPING.keys():
            if f'csi_{key}' in self.file_name:
                self.file_structure = FILENAME_STRUCTURE_MAPPING[key]
                break

        if self.file_structure is None:
            raise ValueError(f"Unknown file structure for {self.file_name}")

    @staticmethod
    def can_read(self, path: str) -> bool:
        """
        Checks if the reader can read the given file path.

        :param self:
        :param path: The file path to check.
        :return: True if the reader can read the file, False otherwise.
        """
        file_name_matched = any(
            str(key) in path for key in FILENAME_STRUCTURE_MAPPING.keys())
        txt_ended = path.endswith('.txt')
        return file_name_matched and txt_ended

    def get_record_length(self) -> int:
        """
        Returns the record length for the file structure.

        :return: The record length.
        """
        return self.file_structure['record_length']

    def get_ts_count(self) -> int:
        """
        Returns the timestamp count for the file structure.

        :return: The timestamp count.
        """
        return self.file_structure['ts_count']

    def get_rssi_count(self) -> int:
        """
        Returns the RSSI count for the file structure.

        :return: The RSSI count.
        """
        return self.file_structure['rssi_count']

    def get_mcs_count(self) -> int:
        """
        Returns the MCS count for the file structure.

        :return: The MCS count.
        """
        return self.file_structure['mcs_count']

    def get_gain_count(self) -> int:
        """
        Returns the gain count for the file structure.

        :return: The gain count.
        """
        return self.file_structure['gain_count']

    def get_csi_count(self) -> int:
        """
        Returns the CSI count for the file structure.

        :return: The CSI count.
        """
        return self.file_structure['csi_count']

    def get_num_subcarriers(self) -> int:
        """
        Returns the number of subcarriers for the file structure.

        :return: The number of subcarriers.
        """
        return self.file_structure['csi_num_subcarriers']

    def get_num_antennas(self) -> int:
        """
        Returns the number of antennas for the file structure.

        :return: The number of antennas.
        """
        return self.file_structure['csi_num_antennas']

    def get_target_fs(self) -> float:
        """
        Returns the target sampling frequency for the file structure.

        :return: The target sampling frequency.
        """
        return self.file_structure['target_fs']

    def read_file(self, file_path: str) -> CSIData:
        file_name = os.path.basename(file_path)
        ret_data = CSIData(file_name)
        data = open(file_path, 'r').read().strip().split()
        length = len(data)
        record_length = self.get_record_length()
        n_records = length // record_length
        extra = length % record_length
        if extra != 0:
            print(f"Warning: Incomplete record at the end of file {file_path}")
            data = data[:length - extra]

        for i in range(n_records):
            start = i * record_length
            cur_record = data[start:start + record_length]

            # 1) ts
            ts_count = self.get_ts_count()
            ts_vals = cur_record[0:ts_count]
            ts = [float(x) for x in ts_vals]

            # 2) rssi
            rssi_count = self.get_rssi_count()
            rssi_vals = cur_record[ts_count: ts_count + rssi_count]
            rssi = [float(x) for x in rssi_vals]

            # 3) mcs
            mcs_count = self.get_mcs_count()
            mcs_start = ts_count + rssi_count
            mcs_vals = cur_record[mcs_start: mcs_start + mcs_count]
            mcs = float(mcs_vals[0]) if mcs_count > 0 else 0.

            # 4) gain
            gain_count = self.get_gain_count()
            gain_start = mcs_start + mcs_count
            gain_vals = cur_record[gain_start: gain_start + gain_count]
            gain = [float(x) for x in gain_vals]

            # 5) csi
            csi_count = self.get_csi_count()
            csi_start = gain_start + gain_count
            csi_vals = cur_record[csi_start: csi_start + csi_count]
            # 解析成复数
            csi_list = [_parse_or_zero(v) for v in csi_vals]
            csi_matrix = np.array(csi_list, dtype=complex)

            # reshape => (num_subcarriers, num_antennas)
            n_subcarriers = self.get_num_subcarriers()
            n_antennas = self.get_num_antennas()
            csi_matrix = csi_matrix.reshape(n_subcarriers, n_antennas)

            # fs
            final_fs = self.get_target_fs()

            # frame
            frame = HwOfficeFrame(ts=ts, rssi=rssi, mcs=mcs, gain=gain, csi_matrix=csi_matrix,
                                  num_subcarriers=n_subcarriers, num_antennas=n_antennas, fs=final_fs)
            ret_data.add_frame(frame=frame)

        return ret_data
