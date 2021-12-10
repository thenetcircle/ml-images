import argparse
import os
import pathlib
import sys
from glob import glob
from multiprocessing import Pool

import cv2
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument("-i", "--input", help="Input directory", required=True)
    parser.add_argument("-r", "--replace", help="Replace output files", default=False)
    parser.add_argument("-w", "--workers", help="Number of workers", default=os.cpu_count())
    parser.add_argument(
        "-s", "--stage",
        help="Stage, e.g. 'train,test', else all subdirs will be traversed",
        default=None
    )

    _args = parser.parse_args()

    try:
        n_workers = int(float(_args.workers))
    except TypeError:
        print(f"error: worker argument is not a valid int: {_args.workers}")
        sys.exit(1)

    if n_workers < 1 or n_workers > 128:
        print(f"error: worker amount needs to be in range [1, 128] but was '{n_workers}'")
        sys.exit(1)

    if not os.path.exists(_args.input):
        print(f"error: input dir doesn't exist: {_args.input}")
        sys.exit(1)

    if _args.stage is not None:
        stages = _args.stage.split(",")
        for stage in stages:
            stage_dir = f"{_args.input}/{stage}"
            if not os.path.exists(stage_dir):
                print(f"error: stage dir '{stage_dir}' does not exist")
                sys.exit(1)

    if not os.path.exists(_args.output):
        try:
            pathlib.Path(_args.output).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"error: could not create output dir '{_args.output}': {str(e)}")
            sys.exit(1)

    return _args


def process_image(input_tuple):
    try:
        _file_path, _output_path = input_tuple
        temp_img = cv2.imread(_file_path, cv2.IMREAD_COLOR)
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2Lab)

        l, a, b = cv2.split(img_lab)
        img_l = clahe.apply(l)
        img_clahe = cv2.merge((img_l, a, b))

        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2RGB)
        cv2.imwrite(_output_path, img_clahe)
    except Exception as e1:
        print(f"error: could not process file '{file_path}' because: {str(e1)}")


if __name__ == "__main__":
    args = parse_arguments()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    files = list()

    if args.stage is not None:
        stage_dirs = [f"{args.input}/{stage}" for stage in args.stage.split(",")]
    else:
        stage_dirs = glob(f"{args.input}/*")

    # create a list of files to process and their corresponding output path
    for train_or_test in stage_dirs:
        for label_dir in glob(f"{train_or_test}/*"):
            output_dir = f"{args.output}/{label_dir.split('/', maxsplit=1)[1]}"

            if not os.path.exists(output_dir):
                try:
                    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"error: could not create output dir '{output_dir}': {str(e)}")
                    sys.exit(1)

            for file_path in glob(f"{label_dir}/*.jpg"):
                output_path = f"{args.output}/{file_path.split('/', maxsplit=1)[1]}"

                if os.path.exists(output_path) and not args.replace:
                    continue

                files.append((file_path, output_path))

    # process in parallel and save to disk
    with Pool(processes=int(float(args.workers))) as pool:
        list(tqdm(pool.imap_unordered(process_image, files), total=len(files)))
