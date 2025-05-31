import os
import h5py
from csi.csi_data import CSIData
from csi.frames.wiprox_mat_frame import WiproxMatFrame
from sdp.reader.reader import Reader


class WiproxMatReader(Reader):
    
    """
    Reader for Wiprox .mat files.
    """
    def __init__(self):
        super().__init__()

    @classmethod
    def can_read(cls, file_path: str) -> bool:
        """
        Check if the reader can read the given file path.
        """
        return file_path.endswith('.mat')

    def read_file(self, file_path: str) -> CSIData:
        file_name = os.path.basename(file_path)
        ret_data = CSIData(file_name)

        try:
            # 使用 h5py 读取 .mat 文件 (v7.3)
            with h5py.File(file_path, 'r') as mat_file:
                # 获取数据
                iot_csi_refs = mat_file['iot_csi'][:].T  # shape (n_samples, 1)
                ue_csi_refs = mat_file['ue_csi'][:].T  # shape (n_samples, 1)
                # shape (n_samples, 1)
                distance_refs = mat_file['dist_val'][:].T
                print(f"iot_csi_refs shape: {iot_csi_refs.shape}")
                print(f"ue_csi_refs shape: {ue_csi_refs.shape}")
                print(f"distance_refs shape: {distance_refs.shape}")

                for i in range(len(iot_csi_refs)):
                    iot_csi_ref = iot_csi_refs[i, 0]
                    iot_csi_path = h5py.h5r.get_name(iot_csi_ref, mat_file.id)
                    iot_csi_data = mat_file[iot_csi_path][:].T
                    print(f"iot_csi_data shape : {iot_csi_data.shape}")

                    ue_csi_ref = ue_csi_refs[i, 0]  # Each ue_csi[i] is a cell
                    ue_csi_path = h5py.h5r.get_name(ue_csi_ref, mat_file.id)
                    ue_csi_data = mat_file[ue_csi_path][:].T

                    # Each dist_val[i] is a cell
                    distance_ref = distance_refs[i, 0]
                    distance_path = h5py.h5r.get_name(
                        distance_ref, mat_file.id)
                    distance_data = mat_file[distance_path][:].T
                    frame = WiproxMatFrame(
                        iot_csi_data=iot_csi_data,
                        ue_csi_data=ue_csi_data,
                        distance_data=distance_data
                    )
                    ret_data.add_frame(frame=frame)
        except Exception as e:
            raise ValueError(f"Error loading .mat file: {file_path}, {str(e)}")
        return ret_data
