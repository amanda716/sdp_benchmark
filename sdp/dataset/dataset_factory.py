from sdp.dataset.CSIDateset import CSIDataset
from sdp.dataset.datasets.hw_dataset import HwDataset
from sdp.dataset.datasets.wiprox_dataset import WiProxDataset
from sdp.reader.readers import bfee_reader, hw_office_reader, wiprox_mat_reader


class DatasetFactory:
    # 数据集映射表（读取器类型 -> 数据集类）
    DATASET_MAP = {
        bfee_reader.BfeeReader: CSIDataset,   # only GenDataset by now
        hw_office_reader.HwOfficeReader: HwDataset,
        wiprox_mat_reader.WiproxMatReader: WiProxDataset
    }

    @classmethod
    def create_dataset(cls, process_res, reader):
        """根据Reader类型创建对应的Dataset实例"""
        dataset_cls = cls.DATASET_MAP.get(type(reader))
        if not dataset_cls:
            raise TypeError(f"dataset {type(reader).__name__} not found")
        if dataset_cls is WiProxDataset:
            return dataset_cls(process_res)
        else:
            return dataset_cls(*process_res)
