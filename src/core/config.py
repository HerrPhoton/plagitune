import os
from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).parent.parent.parent
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

    MODEL_CONF = 0.4
    MODEL_IOU = 0.2


settings = Settings()
