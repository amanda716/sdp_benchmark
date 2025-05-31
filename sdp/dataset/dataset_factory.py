from sdp.dataset.datasets import bfee_dataset, hw_dataset, wiprox_dataset
from sdp.dataset.datasets.bfee_dataset import BfeeDataset
from sdp.dataset.datasets.hw_dataset import HwDataset
from sdp.dataset.datasets.wiprox_dataset import WiProxDataset
from sdp.reader.readers import bfee_reader, hw_office_reader, wiprox_mat_reader


class DatasetFactory:
    # 数据集映射表（读取器类型 -> 数据集类）
    DATASET_MAP = {
        bfee_reader.BfeeReader: bfee_dataset.BfeeDataset,
        hw_office_reader.HwOfficeReader: hw_dataset.HwDataset,
        wiprox_mat_reader.WiproxMatReader: wiprox_dataset.WiProxDataset
    }

    @classmethod
    def create_dataset(cls, process_res, reader):
        """根据Reader类型创建对应的Dataset实例"""
        dataset_cls = cls.DATASET_MAP.get(type(reader))
        if not dataset_cls:
            raise TypeError(f"dataset {type(reader).__name__} not found")
        if dataset_cls is HwDataset or dataset_cls is BfeeDataset:
            return dataset_cls(*process_res)
        else:
            return dataset_cls(process_res)
