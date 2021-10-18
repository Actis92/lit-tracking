from pytorch_lightning import LightningDataModule


class TrackingDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
