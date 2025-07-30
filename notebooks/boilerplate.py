import os
import torch


def setup_nb() -> torch.device:
    REPO_NAME = "hubbard-transformer"

    if not os.getcwd().endswith(REPO_NAME):
        os.chdir("..")

    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available. Running on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Changed working dir to", os.getcwd())

    return device
