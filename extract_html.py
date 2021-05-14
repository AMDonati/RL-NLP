from pathlib import Path
from shutil import copyfile
import os
files = Path(r'output/clevr_on_policy_debug_1').rglob('*/*2021*/metrics/*test_test_images_with_trunc*.html')
path_to = Path("output/clevr_on_policy_html")
path_to.mkdir(parents=True, exist_ok=True)
# remove 'r' string if you're on macos.

for file in files:
    parent_1 = file.parent.name
    parent_2 = file.parent.parent.name
    parent_exp = file.parent.parent.parent.name
    path_to_folder = Path(path_to, parent_exp, f"{parent_2}_{file.name}")
    path_to_folder.parent.mkdir(parents=True, exist_ok=True)
    #os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
    copyfile(file, path_to_folder)
    print(f"{file.name} --> {parent_1}_{parent_2}{file.suffix}")

