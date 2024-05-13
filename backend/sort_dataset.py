from os import listdir, rename, makedirs
from os.path import exists
from random import random

def gather_in_folder(folder: str, dataset_dir: str|list[str], add_label: bool) -> None:
    if not exists(folder): makedirs(folder)
    if isinstance(dataset_dir, str):
        dataset_dir = [dataset_dir]
    for datadir in dataset_dir:
        for dir in listdir(datadir):
            full_path = datadir + "/" + dir
            for file in listdir(full_path):
                file_src = full_path + "/" + file
                label = ""
                if add_label:
                    label = dir[-5:]
                destination = folder + "/" + label + file
                rename(file_src, destination)


def relocate_and_split(folder: str, dataset_dir: str, train_test_split: bool = False, validation_rate: float = 0.2) -> None:
    if not exists(folder): raise ValueError("Directory does not exist.")
    if train_test_split:
        makedirs(f"{dataset_dir}/train/fake/", exist_ok=True)
        makedirs(f"{dataset_dir}/train/real/", exist_ok=True)
        makedirs(f"{dataset_dir}/test/fake/", exist_ok=True)
        makedirs(f"{dataset_dir}/test/real/", exist_ok=True)
        for file in listdir(folder):
            file_src = folder + "/" + file
            if random() > validation_rate:
                if file.startswith("fake"):
                    rename(file_src, f"{dataset_dir}/train/fake/" + file)
                elif file.startswith("real"):
                    rename(file_src, f"{dataset_dir}/train/real/" + file)
            else:
                if file.startswith("fake"):
                    rename(file_src, f"{dataset_dir}/test/fake/" + file)
                elif file.startswith("real"):
                    rename(file_src, f"{dataset_dir}/test/real/" + file)
    
    else:
        for file in listdir(folder):
            file_src = folder + "/" + file
            if file.startswith("fake"):
                rename(file_src, f"{dataset_dir}/fake/" + file)
            elif file.startswith("real"):
                rename(file_src, f"{dataset_dir}/real/" + file)


def main() -> None:
    # gather_in_folder("dataset2/all", ["dataset2/separated/train", "dataset2/separated/test"], False)
    relocate_and_split("dataset2/all", "dataset2/separated", train_test_split=True)

if __name__ == "__main__":
    main()