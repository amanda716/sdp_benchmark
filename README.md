# Terms

SDP - Sensing Dataset Platform
* RSSI: Received Signal Strength Indicator
* MCS: Modulation and Coding Scheme
* dB: Decibel
* CSI: Channel State Information


## Data Format

### Intel 5300 Wifi card
The .dat file has raw data from Intel 5300 Wifi signal, specifically CSI and other related metrics. It is binary format data file. 


## Dataset
### Wi-Prox dataset (Suggesting proximity or distance)
Based on the variable names and the context of loading data from a "Wi-Prox" dataset, here are the likely meanings of iot_csi, ue_csi, and dist_val:

iot_csi: This likely refers to Internet of Things Channel State Information. In wireless communication, Channel State Information (CSI) describes how a signal propagates from a transmitter to a receiver. In the context of an IoT-related dataset, this would be the CSI data collected by or related to an IoT device.

ue_csi: This likely refers to User Equipment Channel State Information. User Equipment (UE) is a term used in telecommunications to refer to devices used by end-users to communicate, such as smartphones, tablets, or other wireless devices. So, this would be the CSI data collected by or related to a User Equipment device.

dist_val: This likely refers to Distance Value. Given that the dataset is named "Wi-Prox" (suggesting proximity or distance) and the code is potentially used for positioning or tracking tasks, dist_val most likely represents the measured or ground truth distance between the IoT device and the User Equipment for each data sample.

Therefore, your interpretation is likely correct: iot_csi relates to the Internet of Things, and dist_val represents distance.


