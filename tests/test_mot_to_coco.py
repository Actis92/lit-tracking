import json
import os

from lit_tracking.utils.mot_to_coco import Mot20ToCoco

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test(tmpdir):
    converter = Mot20ToCoco(input_path=f"{TEST_DIR}/data/mot20", output_path=str(tmpdir))
    converter.convert()
    with open(f"{str(tmpdir)}/train/labels.json") as json_file:
        actual_data = json.load(json_file)
    with open(f"{TEST_DIR}/data/coco/train/labels.json") as json_file:
        expected_data = json.load(json_file)
    assert actual_data == expected_data
