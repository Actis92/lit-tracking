import configparser
import json
from enum import Enum
from pathlib import Path
import os
from os.path import exists
from typing import List, Dict, Tuple

import numpy as np


class Mot20Columns(Enum):
    frame_number = 0  # Indicate at which frame the object is present
    identity_number = 1  # Each pedestrian trajectory is identified by a unique ID (−1 for detections)
    bbox_left = 2  # Coordinate of the top-left corner of the pedestrian bounding box
    bbox_top = 3  # Coordinate of the top-left corner of the pedestrian bounding box
    bbox_width = 4  # Width in pixels of the pedestrian bounding box
    bbox_height = 5  # Height in pixels of the pedestrian bounding box
    confidence_score = 6  # In Det file indicates how confident the detector
    # is that this instance is a pedestrian. In Gt file It acts as a flag whether the entry
    # is to be considered (1) or ignored (0).
    class_name = 7  # Indicates the type of object annotated
    visibility = 8  # Visibility ratio, a number between 0 and 1 that says how much of that object is visible


class Mot20Labels(Enum):
    pedestrian = 1
    person_on_vehicle = 2
    car = 3
    bicycle = 4
    motorbike = 5
    non_motorized_vehicle = 6
    static_person = 7
    distractor = 8
    occluder = 9
    occluder_on_ground = 10
    occluder_full = 11
    reflection = 12
    crowd = 13


class Mot20LabelsSupercategories(Enum):
    pedestrian = "person"
    person_on_vehicle = "person"
    car = "vehicle"
    bicycle = "vehicle"
    motorbike = "vehicle"
    non_motorized_vehicle = "vehicle"
    static_person = "person"
    distractor = "person"
    occluder = "occluder"
    occluder_on_ground = "occluder"
    occluder_full = "occluder"
    reflection = "person"
    crowd = "crowd"


class Mot20Config:
    """This class parse the mot configuration inside the file seqinfo.ini

    :param config_path: a string with the path of the file seqinfo.ini
    """
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.name: str = self.config["Sequence"]["name"]
        self.image_dir: str = self.config["Sequence"]["ImDir"]
        self.frame_rate: int = int(self.config["Sequence"]["frameRate"])
        self.seq_length: int = int(self.config["Sequence"]["seqLength"])
        self.image_width: int = int(self.config["Sequence"]["imWidth"])
        self.image_height: int = int(self.config["Sequence"]["imHeight"])
        self.image_extension: str = self.config["Sequence"]["imExt"]


class Mot20ToCoco:
    """
    This class convert annotations from MOT format to COCO

    :param input_path: path of the data to process
    :param output_path: where save the coco annotation file
    """
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.config_name = "seqinfo.ini"
        self.det_path = "det/det.txt"
        self.gt_path = "gt/gt.txt"

    def convert(self):
        """Create a coco annotation file from the mot20 files
        """
        # for each split create output directory if not exist
        for split in os.listdir(self.input_path):
            Path(f"{self.output_path}/{split}").mkdir(parents=True, exist_ok=True)
            image_cnt = 0
            ann_cnt = 0
            video_cnt = 1
            tid_curr = 0
            tid_last = -1
            out = {'images': [], 'annotations': [], 'videos': [],
                   'categories': []}
            seqs = os.listdir(f"{self.input_path}/{split}")
            out["categories"] = self.convert_labels_to_coco_categories()
            for seq in sorted(seqs):
                config_path = f"{self.input_path}/{split}/{seq}/{self.config_name}"
                if exists(config_path):
                    config = Mot20Config(config_path)
                    out['videos'].append({'id': video_cnt, 'file_name': seq})
                    out["images"] = self.extract_image_info(config=config,
                                                            seq=seq, video_cnt=video_cnt,
                                                            image_cnt=image_cnt)
                    if split != 'test':
                        # in this case the file gt.txt doesn't exist
                        ann_path = f"{self.input_path}/{split}/{seq}/{self.gt_path}"
                        out["annotations"], ann_cnt, tid_curr, tid_last = self.extract_annotations(ann_path=ann_path,
                                                                                                   ann_cnt=ann_cnt,
                                                                                                   tid_curr=tid_curr,
                                                                                                   tid_last=tid_last,
                                                                                                   image_cnt=image_cnt)
                    image_cnt += config.seq_length
                    video_cnt += 1
                print('loaded {} for {} images and {} samples'.format(split,
                                                                      len(out['images']), len(out['annotations'])))
            json.dump(out, open(f"{self.output_path}/{split}/annotations.json", 'w'))

    @staticmethod
    def extract_image_info(config: Mot20Config, seq: str, video_cnt: int, image_cnt: int) -> List[Dict]:
        """Extract all the information regarding the images

        :param config: contains the configuration obtained parsing seqinfo.ini
        :param seq: name of the sequence
        :param video_cnt: counter of the videos already processed
        :param image_cnt: counter of the images already processed
        :return: a list containing the info from each image saved in a dictionary
        """
        out = []
        for i in range(config.seq_length):
            image_cnt += 1
            image_info = {'file_name': '{}/{}/{:06d}{}'.format(seq, config.image_dir, i + 1,
                                                               config.image_extension),  # image name.
                          'id': image_cnt,  # image number in the entire training set.
                          'frame_id': i + 1,
                          # image number in the video sequence, starting from 1.
                          'prev_image_id': image_cnt - 1 if i > 0 else -1,
                          # image number in the entire training set.
                          'next_image_id': image_cnt + 1 if i < config.seq_length - 1 else -1,
                          'video_id': video_cnt,
                          'height': config.image_height,
                          'width': config.image_width}
            out.append(image_info)
        return out

    @staticmethod
    def extract_annotations(ann_path: str, ann_cnt: int, tid_curr: int,
                            tid_last: int, image_cnt: int) -> Tuple[List[Dict], int, int, int]:
        """Extract info from the groudtruth, gt.txt, and create coco annotations

        :param ann_path: path of the gt.txt file with the annotations
        :param ann_cnt: counter of the annotations already processed
        :param tid_curr: current trajectory id
        :param tid_last: last trajectory id
        :param image_cnt: counter of the images already processed
        :return: a list containing the info from each annotation saved in a dictionary
        """
        out = []
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        for i in range(anns.shape[0]):
            frame_id = int(anns[i][Mot20Columns.frame_number.value])
            track_id = int(anns[i][Mot20Columns.identity_number.value])
            cat_id = int(anns[i][Mot20Columns.class_name.value])
            bbox = anns[i][Mot20Columns.bbox_top.value:Mot20Columns.confidence_score.value].tolist()
            ann_cnt += 1
            # check if we are on the same trajectory or if it's changed
            if not track_id == tid_last:
                tid_curr += 1
                tid_last = track_id
            ann = {'id': ann_cnt,
                   'category_id': cat_id,
                   'image_id': image_cnt + frame_id,
                   'track_id': tid_curr,
                   'bbox': bbox,
                   'conf': float(anns[i][Mot20Columns.confidence_score.value]),
                   'iscrowd': 1 if cat_id == Mot20Labels.crowd.value else 0,
                   'area': float(anns[i][Mot20Columns.bbox_width.value] * anns[i][Mot20Columns.bbox_height.value])}
            out.append(ann)
        return out, ann_cnt, tid_curr, tid_last

    @staticmethod
    def convert_labels_to_coco_categories() -> List[Dict]:
        """Convert MOT labels to coco categories

        :return: a list containing the info for each label saved in a dictionary
        """
        output = []
        for label in Mot20Labels:
            output.append({"supercategory": Mot20LabelsSupercategories[label.name].value,
                           "id": label.value,
                           "name": label.name})
        return output
