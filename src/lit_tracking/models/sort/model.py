import icevision

from lit_tracking.tracking_model import TrackingModel


class Sort(TrackingModel):
    def __init__(self, obj_dection_model: icevision.models):
        super().__init__(obj_dection_model)
