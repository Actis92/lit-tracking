from dataclasses import dataclass

import flash
from flash import InputTransform
from flash.image import ObjectDetectionData, ObjectDetector
from icevision.tfms import A

from flash.core.integrations.icevision.transforms import IceVisionTransformAdapter

@dataclass
class ResizeTransform(InputTransform):

    image_size: int = 128

    def per_sample_transform(self):
        return IceVisionTransformAdapter(
            [*A.aug_tfms(size=1024, presize=1920), A.Normalize()]
        )



if __name__ == "__main__":
    datamodule = ObjectDetectionData.from_coco(
        train_folder="../../../tests/data/coco/train/data",
        train_ann_file="../../../tests/data/coco/train/labels.json",
        val_split=0.5,
        batch_size=1,
        train_transform=ResizeTransform,
        val_transform=ResizeTransform
    )

    # 2. Build the task
    model = ObjectDetector(head="efficientdet", backbone="d0", num_classes=datamodule.num_classes,
                           image_size=1024)

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=1)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    predictions = model.predict(
        [
            "../../../tests/data/coco/train/data/MOT20-01-000001.jpg",
        ]
    )
    print(predictions)
