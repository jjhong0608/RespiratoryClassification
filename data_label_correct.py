from pathlib import Path
import pandas as pd

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



if __name__ == '__main__':
    df = pd.read_excel("/Users/jjhong0608/Documents/충남대병원 청진음 데이터/청진음_데이터_정리.xlsx")
    print(df)
    print(df["lung_sound(11)"].value_counts())
    print(df["lung_sound(33)"].value_counts())

    df_11 = df[df["classifier"] == 11]
    df_33 = df[df["classifier"] == 33]
    df_77 = df[df["classifier"] == 77]
    df_77 = df_77[df_77["lung_sound(11)"] == df_77["lung_sound(33)"]]

    print(df_11["lung_sound(11)"].value_counts())
    print(df_33["lung_sound(33)"].value_counts())
    print(df_77["lung_sound(11)"].value_counts())

    normal_11 = df_11[df_11["lung_sound(11)"] == "normal"]
    normal_33 = df_33[df_33["lung_sound(33)"] == "normal"]
    normal_77 = df_77[df_77["lung_sound(11)"] == "normal"]

    normal_df = pd.concat([normal_11, normal_33, normal_77])
    print(normal_df.shape)

    crackle_11 = df_11[df_11["lung_sound(11)"] == "crackle"]
    crackle_33 = df_33[df_33["lung_sound(33)"] == "crackle"]
    crackle_77 = df_77[df_77["lung_sound(11)"] == "crackle"]
    crackle_df = pd.concat([crackle_11, crackle_33, crackle_77])
    print(crackle_df.shape)


    wheeze_11 = df_11[df_11["lung_sound(11)"] == "wheezing"]
    wheeze_33 = df_33[df_33["lung_sound(33)"] == "wheezing"]
    wheeze_77 = df_77[df_77["lung_sound(11)"] == "wheezing"]
    wheeze_df = pd.concat([wheeze_11, wheeze_33, wheeze_77])
    print(wheeze_df.shape)

    rhonchi_11 = df_11[df_11["lung_sound(11)"] == "rhonchi"]
    rhonchi_33 = df_33[df_33["lung_sound(33)"] == "rhonchi"]
    rhonchi_77 = df_77[df_77["lung_sound(11)"] == "rhonchi"]
    rhonchi_df = pd.concat([rhonchi_11, rhonchi_33, rhonchi_77])
    print(rhonchi_df.shape)

    def copy_file(df, src, dst):
        fname = df["file_name"]
        Path(dst).mkdir(parents=True, exist_ok=True)
        for name in fname:
            if (Path(src) / name).with_suffix(".wav").exists():
                (Path(dst) / name).with_suffix(".wav").symlink_to((Path(src) / name).with_suffix(".wav"))
            else:
                logger.warning(f"file {name} not found")

    src = "/Users/jjhong0608/Documents/충남대병원 청진음 데이터/new_audio"
    dst = "/Users/jjhong0608/Documents/AudioData/RespiratoryClassification/DATA/ALL_CNUH_DATA/ALL_DATA"
    copy_file(normal_df, src, dst+"/normal")
    copy_file(crackle_df, src, dst+"/crackle")
    copy_file(wheeze_df, src, dst+"/wheeze")
    copy_file(rhonchi_df, src, dst+"/rhonchi")
