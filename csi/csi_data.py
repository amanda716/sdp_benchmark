class CSIData:
    """
    A class to represent a CSI data object.
    """

    def __init__(self, file_name: str):
        """
        Initializes the CSIData object with the provided data.

        :param file_name: The name of a file with raw data.
        """
        self.file_name = file_name
        self.frames = []

    def add_frame(self, frame):
        """
        Adds a CSI frame to the CSIData object.

        :param frame: The CSI frame to be added.
        """
        self.frames.append(frame)
