import os
import yaml
import torch


class DotNest:
    def __init__(self, path: str):
        dictionary = yaml.safe_load(open(path, "r"))
        for k, v in dictionary.items():
            setattr(self, k, v)


this_dir = os.path.dirname(os.path.abspath(__file__))
SETTING = DotNest(os.path.join(this_dir, "setting.yaml"))
SETTING.SCORE_SAVE_PATH = os.path.join(SETTING.SAVE_PATH, "score.txt")
SETTING.MODEL_SAVE_PATH = os.path.join(SETTING.SAVE_PATH, "model")
SETTING.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
