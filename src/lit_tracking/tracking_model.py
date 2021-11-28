import icevision
from pytorch_lightning import LightningModule


class TrackingModel(LightningModule):
    def __init__(self, obj_dection_model: icevision.models):
        super().__init__()
