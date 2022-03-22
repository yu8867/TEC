import os

import pandas as pd

from CreateDataSet import save_h5_dataset

df_names = [
    "偉大なる夢.h5",
    "影男.h5",
    "怪奇四十面相.h5",
    "悪霊物語.h5",
    # "押絵と旅する男.h5",
    "火星の運河.h5",
    "火繩銃.h5",
    "陰獣.h5",
    "黄金豹.h5"
]

def merge_df(data_dir: str):
    df_list = []

    for i in df_names:
        df_list.append(pd.read_hdf(data_dir + i))

    merged_df = pd.concat(df_list)

    merged_df = merged_df.reset_index(drop=True)

    merged_df.to_hdf("/home/user/TEC/data/dataset/merged_dataset.h5", key="datasets")

    save_h5_dataset("/home/user/TEC/data/dataset")


if __name__ == '__main__':
    os.makedirs("/home/user/TEC/data/dataset", exist_ok=True)
    merge_df("/home/user/TEC/data/anotated/")