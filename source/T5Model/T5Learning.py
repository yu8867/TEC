import random
import numpy as np
import torch
import pytorch_lightning as pl
import sys

from T5Tools.CreateDataSet import save_h5_dataset
from T5Tools.FineTunerModel import T5FineTuner
from T5Tools.T5Parameters import args_dict, args_t5, train_params, PICTURE_MODEL_DIR, CONTEXT_MODEL_DIR


# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def learning(is_picture: bool):
    set_seed(42)
    """ 転移学習を実行 Context Label (1/2)"""
    model = T5FineTuner(args_t5, is_picture)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    if is_picture:
        # 最終エポックのモデルを保存
        model.tokenizer.save_pretrained(CONTEXT_MODEL_DIR)
        model.model.save_pretrained(CONTEXT_MODEL_DIR)
    else:
        model.tokenizer.save_pretrained(PICTURE_MODEL_DIR)
        model.model.save_pretrained(PICTURE_MODEL_DIR)

    del model

if __name__ == '__main__':
    print(sys.argv[1])
    if sys.argv[1] == "1":
        print("Picture Learning")
        learning(True)
    else:
        print("Context Learning")
        learning(False)


