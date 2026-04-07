import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import isaacgym
from utils.runner import Runner

if __name__ == "__main__":
    runner = Runner(test=True)
    runner.play()
