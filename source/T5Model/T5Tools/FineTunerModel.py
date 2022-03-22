"""## 学習処理クラス

[PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)を使って学習します。

PyTorch-Lightningとは、機械学習の典型的な処理を簡潔に書くことができるフレームワークです。
"""
import inspect

import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from T5Tools.DataSetTools import H5Dataset
from T5Tools.CreateDataSet import save_h5_dataset
from torch.utils.data import DataLoader


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, is_picture):
        super().__init__()
        self.save_hyperparameters(hparams, ignore="is_picture")
        # 事前学習済みモデルの読み込み
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        # トークナイザーの読み込み
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path, is_fast=True)

        self.is_picture = is_picture


    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked),
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    # def validation_epoch_end(self, outputs):
    #     """バリデーション完了処理"""
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     self.log("val_loss", avg_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    # def test_epoch_end(self, outputs):
    #     """テスト完了処理"""
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     self.log("test_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_dataset(self, tokenizer, type_path, args):
        """データセットを作成する"""
        return H5Dataset(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            type_path=type_path,
            input_max_len=args.max_input_length,
            target_max_len=args.max_target_length,
            is_picture=self.is_picture
        )

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            if self.is_picture:
                model_type = "picture"
            else:
                model_type = "contexsts"


            train_dataset = self.get_dataset(tokenizer=self.tokenizer,
                                             type_path="train_{}.h5".format(model_type), args=self.hparams)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer,
                                           type_path="dev_{}.h5".format(model_type), args=self.hparams)
            self.val_dataset = val_dataset

            self.t_total = (
                    (len(train_dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                    // self.hparams.gradient_accumulation_steps
                    * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.train_batch_size,
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.eval_batch_size,
                          num_workers=4)