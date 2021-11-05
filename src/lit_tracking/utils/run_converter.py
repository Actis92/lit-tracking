import argparse
import importlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Path that contains data in MOT format', required=True)
    parser.add_argument('--output_path', help='Path that will contains the output', required=True)
    parser.add_argument('--converter_name', help='Name of the converter to use', required=True)
    args = parser.parse_args()
    module = importlib.import_module("lit_tracking.converter.mot_to_coco")
    mot2coco = getattr(module, args.converter_name)(input_path=args.input_path, output_path=args.output_path)
    mot2coco.convert()
