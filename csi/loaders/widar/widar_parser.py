
from typing import Optional
import numpy as np
import struct
import logging
from bitstring import BitArray


def parse_as_bfee_record(payload: bytes) -> Optional[dict]:
    """
    Parse a byte array into a bfee (Boamforming Feedback) record.

    Args:
        data (bytes): The byte array to parse.

    Returns:
        Optional[dict]: A dictionary containing the parsed bfee record, or None if parsing fails.
    """
    timestamp_low = int.from_bytes(payload[0:4], 'little')
    bfee_count = int.from_bytes(payload[4:6], 'little')
    n_receive_antennas = int.from_bytes(payload[8], 'little')
    n_transmit_antennas = int.from_bytes(payload[9], 'little')
    rssi_antenna_a = int.from_bytes(payload[10], 'little')
    rssi_antenna_b = int.from_bytes(payload[11], 'little')
    rssi_antenna_c = int.from_bytes(payload[12], 'little')
    noise = struct.unpack('b', payload[13:14])[0]
    auto_gain_control = int.from_bytes(payload[14], 'little')
    selected_antenna = int.from_bytes(payload[15], 'little')
    csi_byte_length = int.from_bytes(payload[16:18], 'little')
    fake_rate = int.from_bytes(payload[18:20], 'little')

    header_length = 20
    n_subcarriers = 30
    bits_per_component = 8
    n_components = 2
    pilot_bits = 3
    n_rx_tx_pairs = n_receive_antennas * n_transmit_antennas
    calculated_byte_length = (n_subcarriers * n_rx_tx_pairs *
                              n_bits_per_component * n_components + pilot_bits) + 7 // 8  # type: ignore

    if csi_byte_length != calculated_byte_length:
        return None
    if len(payload) != csi_byte_length + header_length:
        return None

    csi_bytes = payload[header_length: header_length+csi_byte_length]
    csi_arr = np.zeros((n_subcarriers, n_receive_antennas,
                       n_transmit_antennas), dtype=np.complex64)

    csi_bitstream = BitArray(bytes=csi_bytes)

    bit_index = 0
    for sc_index in range(n_subcarriers):
        bit_index += pilot_bits
        for j in range(n_rx_tx_pairs):
            real8 = csi_bitstream[bit_index:bit_index +
                                  bits_per_component].uint
            imag8 = csi_bitstream[bit_index +
                                  bits_per_component:bit_index + 2 * bits_per_component].uint
            bit_index += 2 * bits_per_component
            if real8 > 127:
                real8 -= 256
            if imag8 > 127:
                imag8 -= 256
            rx_i = j % n_receive_antennas
            tx_i = j // n_receive_antennas
            csi_arr[sc_index, rx_i, tx_i] = np.complex64(real8, imag8)

    # Define the structure of the bfee record
    return {
        'timestamp_low': timestamp_low,
        'bfee_count': bfee_count,
        'n_receive_antennas': n_receive_antennas,
        'n_transmit_antennas': n_transmit_antennas,
        'rssi_antenna_a': rssi_antenna_a,
        'rssi_antenna_b': rssi_antenna_b,
        'rssi_antenna_c': rssi_antenna_c,
        'noise': noise,
        'csi': csi_arr
    }
