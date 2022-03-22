# TEC-制作物紹介
### テーマ
***

<img width="1089" alt="スクリーンショット 2022-03-13 21 26 52" src="https://user-images.githubusercontent.com/95354321/158059315-9c222a4e-3d1f-449c-a57d-1ade2c3b5c73.png">


### 自然言語処理

***
#### アノテーション<br>
&emsp;T5に学習させるために青空文庫から8本の江戸川乱歩作品をデータセットとして使用した。以下の図のように"小説中の名詞がハイライトされている文章が画像を連想させるような語句があるか"、"その文章を画像処理を行い出力として生成された画像は文章を正確に表しているか”の２つをラベル付けした。<br>
  * 一つ目のラベルは、モデルからの出力が人間に理解しやすく、その場面を連想しやすい文章を選択させることを目的としている。<br>
  * 二つ目のラベルは、実際に画像処理に文章をインプットした場合から逆算的に都合の良い文章を選ぶことを目的としている。<br>

図において、文章中でハイライトされている単語は、2つ目のラベルの判別において手助けするために抽出した。抽出方法としては日本語のはwakati、英語はnlpkを使用した。(今回は英語の文章は必要ないので表示しないように作成した。)<br><br>
<img width="600" alt="スクリーンショット 2022-03-13 19 59 08" src="https://user-images.githubusercontent.com/95354321/158058538-5f11f39a-0042-45ad-94af-ed05f2cc7afa.png">

***
#### データセット<br>
&emsp;江戸川乱歩作品10作品アノテーションしたデータセットの特徴量は以下に示す。(900)

|  特徴量  |  説明  |
| ---- | ---- |
|  japanese_texts  |  文章の日本語訳  |
|  english_texts  |  文章の英語訳  |
|  picture_name  |  画像の名前  |
|  path  |  画像のパス  |
|  entity_jp  |  文章の固有名詞(日本語)  |
|  entity_en  |  文章の固有名詞(英語)  |
|  contexts_label  |  テキストラベル  |
|  picture_label  |  画像ラベル  |

***
#### T5(Text-To-Text Transfer Transformer)<br>

<img width="761" alt="スクリーンショット 2022-03-13 21 30 11" src="https://user-images.githubusercontent.com/95354321/158059409-dbe1ac03-d63d-4937-894f-d550896730a3.png">

### 画像処理

***
#### GLIDE

<img width="1006" alt="スクリーンショット 2022-03-14 17 32 49" src="https://user-images.githubusercontent.com/95354321/158134496-a593461d-df3d-4741-9d3f-2a07baf59a9b.png">

### 画像選択
![image](https://user-images.githubusercontent.com/65989721/159242960-e048c5bc-d304-42a8-a740-3a0667515896.png)

## 結果

***
![image](https://user-images.githubusercontent.com/65989721/159243557-4cd114f7-dabc-4e34-8958-041ccd30d196.png)

## 複数の画法

<img width="735" alt="スクリーンショット 2022-03-21 21 53 10" src="https://user-images.githubusercontent.com/95354321/159401753-1760ef30-e112-4f34-9482-a193494c7729.png">


## 引用
[GLIDEの論文](https://arxiv.org/abs/2112.10741)  
[GLIDEのGithub](https://github.com/openai/glide-text2im)
