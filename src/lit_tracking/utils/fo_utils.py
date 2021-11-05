import fiftyone as fo


if __name__ == "__main__":
    # A name for the dataset
    name = "mot20"

    # The directory containing the dataset to import
    dataset_dir = "/Users/lucaactisgrosso/PycharmProjects/lit-tracking/tests/data/coco/train"

    # The type of the dataset being imported
    dataset_type = fo.types.COCODetectionDataset  # for example

    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        name=name,
    )
    session = fo.launch_app()
    print("ciao")