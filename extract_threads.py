from pathlib import Path
from shutil import copytree
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True, default="output/vqa_all/scratch/",
                        help="data folder containing experiments")
    parser.add_argument("-path_to", type=str, required=True, default="output/vqa_all_debug/scratch",
                        help="data to copy to ")
    return parser

def extract(args):

    files = Path(args.path).rglob('*/*/2021*')
    path_to = Path(args.path_to)
    path_to.mkdir(parents=True, exist_ok=True)
    # remove 'r' string if you're on macos.

    for file in files:
        parent_1 = file.parent.name
        parent_2 = file.parent.parent.name
        if parent_2 in ["1", "2", "3", "4", "5"]:
            path_to_folder = Path(path_to, parent_1, f"{parent_2}_{file.name}")
            # path_to_folder.mkdir(parents=True, exist_ok=True)
            copytree(file, path_to_folder)
            print(f"{file.name} --> {parent_1}_{parent_2}{file.suffix}")

if __name__ == '__main__':
    args = get_parser()
    extract(args)

