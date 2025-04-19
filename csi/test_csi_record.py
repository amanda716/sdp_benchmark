import pytest
import json
import os
import tempfile
from csi.csi_record import CSI_Config


@pytest.fixture
def test_config():
    return {"ts_count": 5, "rssi_count": 6, "target_fs": 200}


def test_create_instance(test_config):
    record = CSI_Config.create_config(test_config)
    assert record.ts_count == 5
    assert record.rssi_count == 6
    assert record.target_fs == 200


def test_to_dict(test_config):
    """Test dictionary conversion."""
    record = CSI_Config.create_config(test_config)
    record_dict = record.to_dict()
    assert record_dict["ts_count"] == 5
    assert record_dict["rssi_count"] == 6
    assert record_dict["target_fs"] == 200


def test_load_config_from_file(test_config):
    """Test loading configuration from a JSON file."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False,
                                     suffix=".json") as temp_file:
        json.dump(test_config, temp_file)
        temp_file.flush()
        print(f"json file {temp_file.name}")
        record = CSI_Config.load_config_from_file(temp_file.name)
        assert record is not None
        assert record.ts_count == 5
        assert record.rssi_count == 6
        assert record.target_fs == 200
