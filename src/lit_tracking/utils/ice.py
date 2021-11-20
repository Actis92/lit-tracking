from typing import List

import pytorch_lightning as pl
from icevision import tfms, Dataset, models
from icevision.parsers import COCOBBoxParser
from icevision.core.mask import EncodedRLEs, MaskArray
from icevision.core.record import BaseRecord
from icevision.data.prediction import Prediction
from torch.optim import SGD


def from_icevision_detection(record: "BaseRecord"):
    detection = record.detection

    result = {}

    if hasattr(detection, "bboxes"):
        result["bboxes"] = [
            {
                "xmin": bbox.xmin,
                "ymin": bbox.ymin,
                "width": bbox.width,
                "height": bbox.height,
            }
            for bbox in detection.bboxes
        ]

    if hasattr(detection, "masks"):
        masks = detection.masks

        if isinstance(masks, EncodedRLEs):
            masks = masks.to_mask(record.height, record.width)

        if isinstance(masks, MaskArray):
            result["masks"] = masks.data
        else:
            raise RuntimeError("Masks are expected to be a MaskArray or EncodedRLEs.")

    if hasattr(detection, "keypoints"):
        keypoints = detection.keypoints

        result["keypoints"] = []
        result["keypoints_metadata"] = []

        for keypoint in keypoints:
            keypoints_list = []
            for x, y, v in keypoint.xyv:
                keypoints_list.append(
                    {
                        "x": x,
                        "y": y,
                        "visible": v,
                    }
                )
            result["keypoints"].append(keypoints_list)

            # TODO: Unpack keypoints_metadata
            result["keypoints_metadata"].append(keypoint.metadata)

    if getattr(detection, "label_ids", None) is not None:
        result["labels"] = list(detection.label_ids)

    if getattr(detection, "scores", None) is not None:
        result["scores"] = list(detection.scores)

    return result


def from_icevision_predictions(predictions: List[Prediction]):
    result = []
    for prediction in predictions:
        result.append(from_icevision_detection(prediction.pred))
    return result


if __name__ == "__main__":
    parser = COCOBBoxParser(annotations_filepath="../../../tests/data/coco/train/labels.json",
                            img_dir="../../../tests/data/coco/train/data")
    train_records, valid_records = parser.parse()
    image_size = 1024
    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=1920), tfms.A.Normalize()])
    train_ds = Dataset(train_records, train_tfms)
    model_type = models.ross.efficientdet
    extra_args = {}
    backbone = model_type.backbones.tf_lite0
    # The efficientdet model requires an img_size parameter
    extra_args['img_size'] = image_size
    model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map), **extra_args)
    train_dl = model_type.train_dl(train_ds, batch_size=1, num_workers=0, shuffle=True)

    class LightModel(model_type.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    light_model = LightModel(model)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(light_model, train_dl)

    preds = model_type.predict(light_model, train_ds, detection_threshold=0.01)
    result = from_icevision_predictions(preds)

