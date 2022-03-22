from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn import metrics

from T5Tools.DataSetTools import H5Dataset
from T5Tools.T5Parameters import args_dict, args_t5, train_params, PRETRAINED_MODEL_NAME, PICTURE_MODEL_DIR, CONTEXT_MODEL_DIR, USE_GPU


def test_data_predict(is_picture: bool):
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)
    if is_picture:
        trained_model = T5ForConditionalGeneration.from_pretrained(PICTURE_MODEL_DIR).cuda()
        test_dataset = H5Dataset(tokenizer, args_dict["data_dir"], "test_picture.h5",
                                 input_max_len=args_t5.max_input_length,
                                 target_max_len=args_t5.max_target_length,
                                 is_picture=True
                                 )
    else:
        trained_model = T5ForConditionalGeneration.from_pretrained(CONTEXT_MODEL_DIR).cuda()
        test_dataset = H5Dataset(tokenizer, args_dict["data_dir"], "test_contexsts.h5",
                                 input_max_len=args_t5.max_input_length,
                                 target_max_len=args_t5.max_target_length,
                                 is_picture=False
                                 )


    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4)

    trained_model.eval()

    inputs = []
    outputs = []
    targets = []

    for batch in tqdm(test_loader):
        input_ids = batch['source_ids']
        input_mask = batch['source_mask']

        if USE_GPU:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

        outs = trained_model.generate(input_ids=input_ids,
                                      attention_mask=input_mask,
                                      max_length=args_t5.max_target_length,
                                      return_dict_in_generate=True,
                                      output_scores=True)

        in_dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
                  for ids in input_ids]
        dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
               for ids in outs.sequences]
        target = [tokenizer.decode(ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
                  for ids in batch["target_ids"]]

        inputs.extend(in_dec)
        outputs.extend(dec)
        targets.extend(target)

    """## accuracy"""

    print("predicted, actual, input")

    for index, data in enumerate(inputs):
        print(outputs[index], targets[index], data)

    metrics.accuracy_score(targets, outputs)

    """## ラベル別精度

    [accuracy, precision, recall, f1-scoreの意味](http://ibisforest.org/index.php?F値)
    """

    print(metrics.classification_report(targets, outputs))

if __name__ == '__main__':
    print("Picture")
    test_data_predict(True)
    print("contexsts")
    test_data_predict(False)
