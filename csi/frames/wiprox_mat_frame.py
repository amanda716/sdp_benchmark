from csi.csi_frame import CSIFrame

class WiproxMatFrame(CSIFrame):

    __slot__ = [
        "iot_csi_data",
        "ue_csi_data",
        "distance_data"
    ]

    def __init__(self, iot_csi_data, ue_csi_data, distance_data):
        self.iot_csi_data = iot_csi_data
        self.ue_csi_data = ue_csi_data
        self.distance_data = distance_data
