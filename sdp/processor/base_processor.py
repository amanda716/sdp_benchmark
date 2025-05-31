from abc import ABC, abstractmethod
from typing import List

from csi.csi_data import CSIData


class BaseProcessor(ABC):
    @abstractmethod
    def process(self, data: List[CSIData], folder_path):
        pass
