# TEC
制作物紹介は[こちら](/TEC-制作物紹介.md)

# DEMO

## anotate

https://user-images.githubusercontent.com/65435560/159152456-0190197e-5e96-41a8-8011-9d656c908215.mp4

# Requirement

## aozorabunko-extractor

青空文庫の小説をダウンロードするツールは[こちら](https://github.com/hppRC/aozorabunko-extractor)。

## library

transformers                 4.4.2

torch                        1.10.2+cu113

pytorch-lightning            1.6.0.dev0

scikit-learn                 1.0.2

## deepL-api
[こちら](https://keikenchi.com/how-to-get-a-free-api-key-for-deepl-translator)のサイトを参考にして、取得してください。

# Usage

<details>
<summary>1. 江戸川乱歩作品の取得</summary>

## aozorabunko-extractor
江戸川乱歩作品のプレーンテキストのみを抽出する。
  
```bash
# download data
curl -OL https://github.com/aozorahack/aozorabunko_text/archive/master.zip
unzip master.zip

# install cli
pip install aozorabunko-extractor

# run this command
aozorabunko-extractor --input_dir aozorabunko_text-master/cards/001779 --output_dir data/inputs --break_sentence --min_chars 3

```
## text mining
翻訳しない作品は削除してください。また、6行目のAPI_KEYにdeepLのapi-keyを設定してください。
```bash
python source/Text-mining/ja_to_en.py
```
</details>

<details>
<summary>2. T5モデルによる分類</summary>
  
  英語と日本語が対になった専用データを生成
  ```bash
  python source/T5Model/title_converter.py
  ```
  
  T5Classification.pyの20行目と80行目をラベル付けしたいタイトルに変更
  ```bash
  python source/T5Model/T5Classification.py
  ```
  
 </details>
 
 <details>
<summary>3. 文書抽出・画像生成</summary>
 
  ラベル付け結果やTF-IDFにより画像化可能な文章を厳選後，画像化
  
  source/Extract/main.ipynb を実行
  
 </details>

 <details>
<summary>4. 画像選択</summary>
 
  画像生成した画像を選択
  
  （画像を生成してなかったらsource/Select/extract_generate.ipynbを実行して画像生成。）  
  source/Select/image_selector.ipynbを実行して画像を選択。
  
 </details>
 
-------------------------------
## Anotation and Training

### 前準備
英語から日本語への翻訳
(data/inputsフォルダ内の日本語文章をdata/outputsフォルダへ英訳)
```bash
python source/Text_mining/ja_to_en.py
```
英文から画像を生成(アノテート用)
```bash
python source/Anotate/convert_img.py
```
### Anotation
```bash
source/Anotate/main.ipynb
```
を実行してアノテート
(アノテート結果はdata/hdf5, data/csvフォルダへ出力)

### Model Training

アノテートしたh5ファイルをdata/anotatedフォルダへコピー
一つのh5ファイルへマージ&データセット作成
```bash
python source/T5Model/T5Tools/MergeDF.py
```
学習の実行(picture, context両方)
```bash
python source/T5Model/T5RunLearning.py
```
 <details>
<summary>学習の実行（単体）</summary>
  
argument:0 : ラベルpictureの学習

argument:1 : ラベルcontextの学習（学習のみ）
```bash
python source/T5Model/T5Learning {0|1}
```
 </details>
 
  <details>
<summary>パラメータの調整</summary>
  
```bash
source/T5Model/T5Tools/T5Parameters.py
```
  に記述
  
 PRETRAINED_MODEL_NAMEのコメントアウトで日本語，英語の切り替え
  
  切り替えた場合は
  
  source/T5Model/T5Tools/DataSetTools.py
  
  の43,44行目のコメントアウトで日本語，英語に合わせる
  ```python
  body = row['English_texts']  # En
  # body = row['japanese_texts']  # JP
  ```
  推定も同様に102行目のコメントアウトで切り替え
  
 </details>

# Note

aozorabunko-extractorのdownload dataは大容量のため、通信環境の良い場所で実行してください。
