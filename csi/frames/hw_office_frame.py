from csi.csi_frame import CSIFrame


class HwOfficeFrame(CSIFrame):
    """
    Hardware Office Frame
    """
    slots = [
        "ts",
        "fs",
        "rssi",
        "mcs",
        "gain",
        "csi_matrix",
        "num_subcarriers",
        "num_antennas"
    ]

    def __init__(self, ts, rssi, mcs, gain, csi_matrix, num_subcarriers, num_antennas, fs):
        """
        Initializes the HwOfficeFrame with the provided parameters.

        :param ts: Timestamp data.
        :param rssi: RSSI data.
        :param mcs: MCS data.
        :param gain: Gain data.
        :param csi_matrix: CSI data.
        :param num_subcarriers: Number of subcarriers.
        :param num_antennas: Number of antennas.
        """
        super().__init__()
        self.ts = ts
        self.fs = fs
        self.rssi = rssi
        self.mcs = mcs
        self.gain = gain
        self.csi_matrix = csi_matrix
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
    