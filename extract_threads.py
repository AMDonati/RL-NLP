from pathlib import Path
from shutil import copytree

files = Path(r'output/clevr_on_policy_ext').rglob('*/*/2021*')
path_to = Path("output/clevr_on_policy_ext_debug")
path_to.mkdir(parents=True, exist_ok=True)
# remove 'r' string if you're on macos.

for file in files:
    parent_1 = file.parent.name
    parent_2 = file.parent.parent.name
    if parent_2 in ["1", "2", "3", "4", "5"]:
        path_to_folder = Path(path_to, parent_1, f"{parent_2}_{file.name}")
        #path_to_folder.mkdir(parents=True, exist_ok=True)
        copytree(file, path_to_folder)
        print(f"{file.name} --> {parent_1}_{parent_2}{file.suffix}")
