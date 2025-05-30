from abc import ABC, abstractmethod
from csi.csi_data import CSIData


class Reader(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def can_read(self, file_path: str) -> bool:
        pass

    @abstractmethod
    def read_file(self, file_path: str) -> CSIData:
        pass
