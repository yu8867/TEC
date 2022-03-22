import os

import pandas
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class H5Dataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512, is_picture=True):
        self.file_path = os.path.join(data_dir, type_path)

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.is_picture = is_picture

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, body, image_id):
        # ニュース分類タスク用の入出力形式に変換する。
        input = f"{body}"
        target = f"{image_id}"
        return input, target

    def _build(self):
        merged_df: pandas.DataFrame = pd.read_hdf(self.file_path)

        for index, row in merged_df.iterrows():
            body = row['English_texts']  # En
            # body = row['japanese_texts']  # JP

            if self.is_picture:
                image_id = str(row['picture_label'])  # context
            else:
                image_id = str(row['contexsts_label'])  # context


            input, target = self._make_record(body, image_id)

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input], max_length=self.input_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.target_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


class ServeDataset(Dataset):
    def __init__(self, tokenizer, df: pd.DataFrame, input_max_len=512, target_max_len=512):

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.df = df

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, body, image_id):
        # ニュース分類タスク用の入出力形式に変換する。
        input = f"{body}"
        target = f"{image_id}"
        return input, target

    def _build(self):
        for index, row in self.df.iterrows():
            body = row['english_texts']  # En
            # body = row['japanese_texts']  # JP

            image_id = str(0)
            # image_id = str(row['picture_label']) #context
            # image_id = str(row['contexsts_label']) #context


            input, target = self._make_record(body, image_id)

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input], max_length=self.input_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.target_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
