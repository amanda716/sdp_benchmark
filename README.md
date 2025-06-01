### Overview

- [Terms](#terms)
- [Project Structure Overview](#-project-structure-overview)
- [QuickStart](#quick-start)
- [Extension](#extension)

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

#### `data/` - folder where data is put
* **Attention**: data files in the same sampling project`(e.g. widar, wi-prox)` should be placed under in the same sub_folder, or some files will be ignored or raise exceptions

#### `sdp/` – Platform Core Functionalities

* **Purpose**: Provides essential functionalities for the platform.
* **Subdirectories**:

  * `reader/`: Responsible for reading different data sources and converting them into `CSIData`. This module is designed to be extensible for supporting additional data sources.
  * `utils/`: Contains common utility functions used across the platform, such as logging and other shared services.
  * `processor/`: Processors which process `CSIData` base on the type of `CSIFrame` in the `CSIData` 
  * `dataset/`: Python Classes where the output of processors are stored

#### `handle_format_utils/` - scripts to cope with format problem

This structure promotes modularity and scalability, allowing for easy integration of new data sources and functionalities.

### Quick Start

* prepare for requirement
```
  pip install -r requirements.txt
```

* execute each code block of the `main.ipynb` sequentially

### Extension
1. realize your reader which extends the base class in `reader.py` mainly method `can_read` and `read_file`.
2. realize your processor which extends the base class in `base_processor.py`, mainly method `process`.
3. realize your dataset which extends the base class in `base_dataset.py`.
4. register your class in factories
5. finally, write the param you need and `elif` in `main.ipynb`
```python
# data path
folder_path = PROJECT_ROOT / "data/hw_data"
# param for BfeeProcessor(widar or gait)
task_type = 'Activity Recognition'
final_fs = 1000
# param for Wi_prox_Processor
num_samples = 100000

# add param there
```

```python
# add elif here
global res
if type(processor) == HwProcessor:
    res = list(processor.process(csi_data_list, folder_path=folder_path))
elif type(processor) == BfeeProcessor:
    res = processor.process(csi_data_list, folder_path=folder_path, task_type=task_type, final_fs=final_fs)
elif type(processor) == WiproxProcessor:
    res = processor.process(csi_data_list, folder_path=folder_path, num_samples=num_samples)
```


