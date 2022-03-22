import glob
import random
import pandas as pd


def to_line(data):
    body = data["body"]
    genre_id = data["image_id"]

    assert len(body) > 0
    return f"{body}\t{genre_id}\n"


def load_raw_dataset(data_dir: str):
    false_ratio = 1.0  # Eg. 1.0 = 1:1, 2.0 = 2:1 false:true

    merged_df = pd.read_hdf(data_dir + "/merged_dataset.h5")

    contexsts_label_0 = merged_df.query('contexsts_label == 0').reset_index(drop=True)
    contexsts_label_1 = merged_df.query('contexsts_label == 1').reset_index(drop=True)

    contexsts_label_0_vol = int(len(contexsts_label_1) * false_ratio)

    contexsts_label_0 = contexsts_label_0.sample(frac=1).reset_index(drop=True)
    # contexsts_label_0 = contexsts_label_0.loc[:contexsts_label_0_vol, :]
    dataset_contexsts = pd.concat([contexsts_label_0, contexsts_label_1])
    dataset_contexsts = dataset_contexsts.sample(frac=1).reset_index(drop=True)

    print(len(dataset_contexsts))

    picture_label_0 = merged_df.query('picture_label == 0').reset_index(drop=True)
    picture_label_1 = merged_df.query('picture_label == 1').reset_index(drop=True)

    picture_label_0_vol = int(len(picture_label_1) * false_ratio)
    picture_label_0 = picture_label_0.sample(frac=1).reset_index(drop=True)
    picture_label_0 = picture_label_0.loc[:picture_label_0_vol, :]
    dataset_picture = pd.concat([picture_label_0, picture_label_1])
    dataset_picture = dataset_picture.sample(frac=1).reset_index(drop=True)

    return dataset_picture, dataset_contexsts


"""## データ分割

データセットを70% : 15%: 15% の比率でtrain/dev/testに分割します。

* trainデータ: 学習に利用するデータ
* devデータ: 学習中の精度評価等に利用するデータ
* testデータ: 学習結果のモデルの精度評価に利用するデータ
"""
def save_h5_dataset(data_dir: str):
    all_data_picture, all_data_contexsts = load_raw_dataset(data_dir)
    print(len(all_data_picture), len(all_data_contexsts))

    #データをシャッフル
    random.seed(1234)
    all_data_picture.sample(frac=1, random_state=1234).reset_index(drop=True)
    all_data_contexsts.sample(frac=1, random_state=1234).reset_index(drop=True)

    for i in ["picture", "contexsts"]:
        if i == "picture":
            all_data = all_data_picture
            print("Picture Gen")
        else:
            all_data = all_data_contexsts
            print("Context Gen")

        data_size = len(all_data)
        train_ratio, dev_ratio, test_ratio = 0.7, 0.15, 0.15

        train_vol, dev_vol, test_vol = int(train_ratio * data_size), int(dev_ratio * data_size), int(test_ratio * data_size)

        f_train = all_data.loc[:train_vol, :]
        f_dev = all_data.loc[train_vol:train_vol + dev_vol, :]
        f_test = all_data.loc[train_vol + dev_vol:, :]

        print(len(f_train), len(f_dev), len(f_test))

        f_train.to_hdf(data_dir + "/train_{}.h5".format(i), key="datasets")
        f_dev.to_hdf(data_dir + "/dev_{}.h5".format(i), key="datasets")
        f_test.to_hdf(data_dir + "/test_{}.h5".format(i), key="datasets")

