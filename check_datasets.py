import logging
from pathlib import Path

from rich.logging import RichHandler

logger = logging.getLogger(__name__)
handler = RichHandler(
    rich_tracebacks=True,
    show_path=True,
    omit_repeated_times=False,
)
formatter = logging.Formatter("%(funcName)s - %(message)s")
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.DEBUG)
logging.root.handlers.clear()


SOURCE_DIR = Path("datasets/raw").resolve().expanduser()
AIRWAY_DIR = SOURCE_DIR / "Airway_disease_group"
PALENCHYMAL_DIR = SOURCE_DIR / "Lung_parenchymal_disease_group"
TARGET_DIR = Path("datasets/colored_folds/test").resolve().expanduser()
TARGET_AIRWAY_DIR = TARGET_DIR / "Airway_disease_group"
TARGET_PALENCHYMAL_DIR = TARGET_DIR / "Lung_parenchymal_disease_group"

if __name__ == "__main__":
    if TARGET_DIR.exists():
        logger.info(f"TARGET_DIR: {TARGET_DIR}")
    if TARGET_AIRWAY_DIR.exists():
        logger.info(f"TARGET_AIRWAY_DIR: {TARGET_AIRWAY_DIR}")
    if TARGET_PALENCHYMAL_DIR.exists():
        logger.info(f"PALENCHYMAL_DIR: {PALENCHYMAL_DIR}")
    if SOURCE_DIR.exists():
        logger.info(f"SOURCE_DIR: {SOURCE_DIR}")
    if AIRWAY_DIR.exists():
        logger.info(f"AIRWAY_DIR: {AIRWAY_DIR}")
    if PALENCHYMAL_DIR.exists():
        logger.info(f"PALENCHYMAL_DIR: {PALENCHYMAL_DIR}")

    for file in TARGET_AIRWAY_DIR.glob("*.wav"):
        logger.info(f"Processing {AIRWAY_DIR.joinpath(file.name)}")
        if AIRWAY_DIR.joinpath(file.name).exists():
            logger.info(f" - {AIRWAY_DIR.joinpath(file.name)} is deleted")
            AIRWAY_DIR.joinpath(file.name).unlink()

    for file in TARGET_PALENCHYMAL_DIR.glob("*.wav"):
        logger.info(f"Processing {PALENCHYMAL_DIR.joinpath(file.name)}")
        if PALENCHYMAL_DIR.joinpath(file.name).exists():
            logger.info(f" - {PALENCHYMAL_DIR.joinpath(file.name)} is deleted")
            PALENCHYMAL_DIR.joinpath(file.name).unlink()
