
class CSIDataLoader:
    def __init__(self, data):
        self.data = data
        self.records = []
        self.record_length = 10
    
    def read_records_from_folder(self, folder: str):
        file_data = []
        