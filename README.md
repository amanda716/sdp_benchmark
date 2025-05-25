### Terms

SDP - Sensing Dataset Platform
* RSSI: Received Signal Strength Indicator
* MCS: Modulation and Coding Scheme
* dB: Decibel
* CSI: Channel State Information


### 📁 Project Structure Overview

#### `csi/` – Channel State Information (CSI) Data Abstraction

* **Purpose**: Handles various data sources and formats by abstracting them into a unified structure called `CSIData`.
* **Components**:

  * `CSIData`: Encapsulates metadata and the actual data frames.
  * `CSIFrame`: Represents the concrete data frames within `CSIData`.

#### `sdp/` – Platform Core Functionalities

* **Purpose**: Provides essential functionalities for the platform.
* **Subdirectories**:

  * `reader/`: Responsible for reading different data sources and converting them into `CSIData`. This module is designed to be extensible for supporting additional data sources.
  * `utils/`: Contains common utility functions used across the platform, such as logging and other shared services.


This structure promotes modularity and scalability, allowing for easy integration of new data sources and functionalities.


