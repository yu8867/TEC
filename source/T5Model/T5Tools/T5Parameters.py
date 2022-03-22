import torch
import argparse

PICTURE_MODEL_DIR = "data/T5-model/picture_model"
CONTEXT_MODEL_DIR = "data/T5-model/context_model"

# PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"
PRETRAINED_MODEL_NAME = "t5-base"
USE_GPU = torch.cuda.is_available()
# USE_GPU = False
# 各種ハイパーパラメータ
args_dict = dict(
    data_dir="/home/user/TEC/data/dataset",  # データセットのディレクトリ
    model_name_or_path=PRETRAINED_MODEL_NAME,
    tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)

args_dict.update({
    "max_input_length": 256,  # 入力文の最大トークン数
    "max_target_length": 4,  # 出力文の最大トークン数
    "train_batch_size": 8,
    "eval_batch_size": 4,
    "num_train_epochs": 8,
})
args_t5 = argparse.Namespace(**args_dict)
train_params = dict(
    accumulate_grad_batches=args_t5.gradient_accumulation_steps,
    gpus=2,
    strategy="ddp",
    # gpus=args_t5.n_gpu,
    max_epochs=args_t5.num_train_epochs,
    precision=16 if args_t5.fp_16 else 32,
    amp_level=args_t5.opt_level,
    gradient_clip_val=args_t5.max_grad_norm,
    amp_backend='apex'
    # checkpoint_callback=checkpoint_callback,
)