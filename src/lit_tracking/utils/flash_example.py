import flash
from flash.image import ObjectDetectionData, ObjectDetector
from icevision.tfms import A

from flash.core.integrations.icevision.transforms import IceVisionTransformAdapter


if __name__ == "__main__":
    train_transform = {
        "pre_tensor_transform": IceVisionTransformAdapter(
            [*A.aug_tfms(size=(1024,896), presize=(1920, 1080)), A.Normalize()]
        )
    }
    datamodule = ObjectDetectionData.from_coco(
        train_folder="/Users/lucaactisgrosso/PycharmProjects/lit-tracking/tests/data/coco/train/data",
        train_ann_file="/Users/lucaactisgrosso/PycharmProjects/lit-tracking/tests/data/coco/train/labels.json",
        val_split=0.5,
        image_size=(1024,896),
        train_transform=train_transform
    )

    # 2. Build the task
    model = ObjectDetector(head="efficientdet", backbone="d0", num_classes=datamodule.num_classes,
                           image_size=(1024,896))

    # 3. Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=1)
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    predictions = model.predict(
        [
            "/Users/lucaactisgrosso/PycharmProjects/lit-tracking/tests/data/coco/train/data/MOT20-01-000001.jpg",
        ]
    )
    print(predictions)