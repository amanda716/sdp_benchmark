class CSIData:
    """
    A class to represent a CSI data object.
    """

    def __init__(self, file_name: str):
        """
        Initializes the CSIData object with the provided data.

        :param data: The data to be stored in the CSIData object.
        """
        self.file_name = file_name
        self.frames = []

    def add_frame(self, frame):
        """
        Adds a CSI frame to the CSIData object.

        :param frame: The CSI frame to be added.
        """
        self.frames.append(frame)