# -*- coding: utf-8 -*-
from glob import glob
from pathlib import Path
import deepl

API_KEY = ''
source_lang = 'JA'
target_lang = 'EN-GB'

translator = deepl.Translator(API_KEY)

outputs = Path('/workspace/data/outputs')
outputs.mkdir(exist_ok=True, parents=True)

txts = glob('/workspace/data/inputs/*.txt')

for txt in txts:
    p_txt = Path(txt)
    title = p_txt.name

    # txtを開いて改行区切りで文字列をリストに代入
    with p_txt.open(mode='r') as f:
        input_txt = f.read().split('\n')

    # すでに翻訳済みの作品はスキップ
    if (outputs/title).exists():
        continue

    # 1行ずつ翻訳し出力
    with (outputs/title).open(mode='a') as f:
        for x in input_txt:
            result = translator.translate_text(
                x, source_lang=source_lang, target_lang=target_lang)
            f.write(str(result)+'\n')
