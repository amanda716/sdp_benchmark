import numpy as np

from csi.csi_frame import CSIFrame


class BfeeFrame(CSIFrame):
    """
    Represents a WiDAR Bfee frame.
    """

    def __init__(self, timestamp_low, bfee_count, n_rx, n_tx,
                 rssi_a, rssi_b, rssi_c, noise, csi_array, agc, antenna_sel, fake_rate):
        super().__init__()
        self.timestamp_low = timestamp_low
        self.bfee_count = bfee_count
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.rssi_a = rssi_a
        self.rssi_b = rssi_b
        self.rssi_c = rssi_c
        self.noise = noise
        self.csi_array = csi_array
        self.agc = agc
        self.antenna_sel = antenna_sel
        self.fake_rate = fake_rate
