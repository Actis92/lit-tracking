import json

from src.lit_tracking.mot_to_coco import Mot20ToCoco


def test(tmpdir):
    converter = Mot20ToCoco(input_path="./data/mot20", output_path=str(tmpdir))
    converter.convert()
    #COCO(annotation_file=f"{str(tmpdir)}/train/annotations.json")
    with open(f"{str(tmpdir)}/train/annotations.json") as json_file:
        actual_data = json.load(json_file)
    with open("./data/mot20/train/annotations.json") as json_file:
        expected_data = json.load(json_file)
    assert actual_data == expected_data