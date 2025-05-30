from sdp.reader.readers import bfee_reader, hw_office_reader, wiprox_mat_reader
from sdp.processor.processors.BfeeProcessor import BfeeProcessor
from sdp.processor.processors.HwProcessor import HwProcessor
from sdp.processor.processors.WiporxProcessor import WiporxProcessor


class ProcessorFactory:
    # 处理器映射表（帧类型 -> 处理器实例）
    PROCESSORS = {
        bfee_reader.BfeeFrame: BfeeProcessor(),
        hw_office_reader.HwOfficeFrame: HwProcessor(),
        wiprox_mat_reader.WiproxMatFrame: WiporxProcessor()
    }

    @classmethod
    def get_processor(cls, frame):
        processor = cls.PROCESSORS.get(type(frame))
        if not processor:
            raise TypeError(f"No processor found for {type(frame).__name__}")
        return processor
