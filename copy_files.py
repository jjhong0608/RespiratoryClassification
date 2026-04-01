from pathlib import Path
import logging
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

CHECK_DIR = (
    Path(
        "/Users/jjhong0608/documents/audiodata/DataProcessing/audio/LungSoundNAClassification/training_sets_lung_sound/train"
    )
    .resolve()
    .expanduser()
)
CHECKTEST_DIR = (
    Path(
        "/Users/jjhong0608/documents/audiodata/DataProcessing/audio/LungSoundNAClassification/training_sets_lung_sound/test"
    )
    .resolve()
    .expanduser()
)
SOURCE_DIR = (
    Path(
        "/Users/jjhong0608/documents/audiodata/DataProcessing/audio/LungSoundNAClassification/lung_sound_data_folds"
        # "/Users/jjhong0608/documents/audiodata/DataProcessing/audio/LungSoundNAClassification/disease_data_folds"
    )
    .resolve()
    .expanduser()
)
TARGET_DIR = (
    Path(
        "/Users/jjhong0608/Documents/AudioData/Whisper/datasets/lung_sound_classification_without_rhonchi2"
        # "/Users/jjhong0608/Documents/AudioData/Whisper/datasets/disease_classification_without_rhonchi"
    )
    .resolve()
    .expanduser()
)

if __name__ == "__main__":
    for file in SOURCE_DIR.rglob("*.wav"):
        label = file.parent.name
        fold = file.parent.parent.name
        if (CHECK_DIR / "rhonchi" / file.name).exists():
            logger.info("rhonchi is NOT copied")
            continue
        if (CHECK_DIR / "crackle+rhonchi" / file.name).exists():
            logger.info("crackle+rhonchi is NOT copied")
            continue
        if (CHECK_DIR / "wheeze+rhonchi" / file.name).exists():
            logger.info("wheeze+rhonchi is NOT copied")
            continue
        if (CHECKTEST_DIR / "rhonchi" / file.name).exists():
            logger.info("rhonchi is NOT copied")
            continue
        if (CHECKTEST_DIR / "crackle+rhonchi" / file.name).exists():
            logger.info("crackle+rhonchi is NOT copied")
            continue
        if (CHECKTEST_DIR / "wheeze+rhonchi" / file.name).exists():
            logger.info("wheeze+rhonchi is NOT copied")
            continue
        (TARGET_DIR / fold / label).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / fold / label / file.name).symlink_to(file)
        logger.info(f"Copied {file} to \n{TARGET_DIR / fold / label / file.name}")
