from sdp.processor.base_processor import BaseProcessor
from csi.frames.bfee_frame import BfeeFrame


class BfeeProcessor(BaseProcessor):
    def process(self, frame: BfeeFrame, folder_path) -> float:
        """处理TXT帧数据"""
        pass
