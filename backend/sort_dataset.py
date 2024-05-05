from os import listdir, rename, makedirs
from os.path import exists

def gather_in_folder(folder: str) -> None:
    if not exists(folder): makedirs(folder)
    dirs = ["dataset/fake/", "dataset/real/"]
    for dir in dirs:
        for file in listdir(dir):
            file_src = dir + "/" + file
            destination = folder + "/" + file
            rename(file_src, destination)


def relocate_and_split(folder: str) -> None:
    if not exists(folder): raise ValueError("Directory does not exist.")
    for file in listdir(folder):
        file_src = folder + "/" + file
        if file.startswith("fake"):
            rename(file_src, "dataset/fake/" + file)
        elif file.startswith("real"):
            rename(file_src, "dataset/real/" + file)


def main() -> None:
    gather_in_folder("dataset/all/")
    # relocate_and_split("dataset/all/")

if __name__ == "__main__":
    main()