from abc import ABC, abstractmethod
from csi.csi_frame import CSIFrame


class BaseProcessor(ABC):
    @abstractmethod
    def process(self, frame: CSIFrame):
        pass
