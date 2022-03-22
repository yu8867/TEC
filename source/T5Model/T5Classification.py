"""# テストデータに対する予測精度を評価"""
import os
import pandas as pd

from tqdm.auto import tqdm
from sklearn import metrics
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)
from T5Tools.DataSetTools import ServeDataset
from torch.utils.data import DataLoader
from source.T5Model.T5Tools.T5Parameters import args_t5, PRETRAINED_MODEL_NAME, USE_GPU, PICTURE_MODEL_DIR, CONTEXT_MODEL_DIR


tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)
trained_picture_model = T5ForConditionalGeneration.from_pretrained(PICTURE_MODEL_DIR).cuda()
trained_context_model = T5ForConditionalGeneration.from_pretrained(CONTEXT_MODEL_DIR).cuda()

df = pd.read_hdf("/home/user/TEC/data/title/押絵と旅する男.h5")

# テストデータの読み込み
test_dataset = ServeDataset(tokenizer, df,
                          input_max_len=args_t5.max_input_length,
                          target_max_len=args_t5.max_target_length)

test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

trained_picture_model.eval()
trained_context_model.eval()

inputs = []
picture_outputs = []
context_outputs = []
confidences = []
targets = []

for index, batch in enumerate(tqdm(test_loader)):
    input_ids = batch['source_ids']
    input_mask = batch['source_mask']

    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    picture_outs = trained_picture_model.generate(input_ids=input_ids,
                                  attention_mask=input_mask,
                                  max_length=args_t5.max_target_length,
                                  return_dict_in_generate=True,
                                  output_scores=True)

    context_outs = trained_context_model.generate(input_ids=input_ids,
                                  attention_mask=input_mask,
                                  max_length=args_t5.max_target_length,
                                  return_dict_in_generate=True,
                                  output_scores=True)

    in_dec = [tokenizer.decode(ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
           for ids in input_ids]
    picture_dec = [tokenizer.decode(ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
           for ids in picture_outs.sequences]
    context_dec = [tokenizer.decode(ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
           for ids in context_outs.sequences]
    # conf = [s.cpu().item() for s in torch.exp(outs.sequences_scores)]
    target = [tokenizer.decode(ids, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
              for ids in batch["target_ids"]]

    inputs.extend(in_dec)
    picture_outputs.extend(picture_dec)
    context_outputs.extend(context_dec)
    # confidences.extend(conf)
    targets.extend(target)


df["picture_label"] = picture_outputs
df["contexsts_label"] = context_outputs

df.to_hdf("/home/user/TEC/data/押絵と旅する男_predicted.h5",key="datasets")


