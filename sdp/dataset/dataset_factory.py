from sdp.dataset.datasets import bfee_dataset, hw_dataset, wiprox_dataset
from sdp.reader.readers import bfee_reader, hw_office_reader, wiprox_mat_reader


class DatasetFactory:
    # 数据集映射表（读取器类型 -> 数据集类）
    DATASET_MAP = {
        bfee_reader.BfeeReader: bfee_dataset.BfeeDataset,
        hw_office_reader.HwOfficeReader: hw_dataset.HwDataset,
        wiprox_mat_reader.WiproxMatReader: wiprox_dataset.WiproxDataset
    }

    @classmethod
    def create_dataset(cls, process_res: tuple, reader):
        """根据Reader类型创建对应的Dataset实例"""
        dataset_cls = cls.DATASET_MAP.get(type(reader))
        if not dataset_cls:
            raise TypeError(f"dataset {type(reader).__name__} not found")
        return dataset_cls(*process_res)
