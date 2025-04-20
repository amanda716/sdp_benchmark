import pytest
from unittest.mock import mock_open, patch
from csi.csi_reader import CSI_Reader
from csi.csi_record import CSI_Config

@pytest.fixture
def mock_config():
    """Fixture to create a mock CSI_Config object."""
    class MockConfig:
        record_length = 10
        ts_count = 2
        rssi_count = 2
        mcs_count = 2
        gain_count = 2
        csi_count = 2
        csi_num_subcarriers = 1
        csi_num_antennas = 2
        target_fs = 5.0

    return MockConfig()


@pytest.fixture
def csi_reader(mock_config):
    """Fixture to create a CSI_Reader instance."""
    return CSI_Reader(config=mock_config)


def test_read_records_complete_records(csi_reader):
    """Test read_records with complete records."""
    mock_file_content = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        records = list(csi_reader.read_records("mock_file.txt"))
        assert len(records) == 2  # Two complete records
        assert records[0]["ts"] == [1.0, 2.0]
        assert records[0]["rssi"] == [3.0, 4.0]
        assert records[1]["ts"] == [11.0, 12.0]
        assert records[1]["rssi"] == [13.0, 14.0]


def test_read_records_incomplete_record(csi_reader, capsys):
    """Test read_records with an incomplete record."""
    mock_file_content = "1 2 3 4 5 6 7 8 9\n"
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        records = list(csi_reader.read_records("mock_file.txt"))
        assert len(records) == 0  # No complete records
        captured = capsys.readouterr()
        assert "Warning: Incomplete record at the end of file" in captured.out


def test_read_records_empty_file(csi_reader):
    """Test read_records with an empty file."""
    mock_file_content = ""
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        records = list(csi_reader.read_records("mock_file.txt"))
        assert len(records) == 0  # No records


def test_read_records_large_file(csi_reader):
    """Test read_records with a large file."""
    mock_file_content = " ".join(str(i) for i in range(1, 101))  # 100 numbers
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        records = list(csi_reader.read_records("mock_file.txt"))
        assert len(records) == 10  # 10 complete records (10 numbers per record)