# -*- coding: utf-8 -*-
from glob import glob
from pathlib import Path
import pandas as pd

entexts = glob('/workspace/data/outputs/*.txt')

for entext in entexts:
    # 日本語のテキストを1行ごとに格納したリストを作成
    file_name = path_entext.name
    with Path(fr'/workspace/data/inputs/{file_name}').open(mode='r') as f:
        l_jatext = f.read().split('\n')

    # 英語のテキストを1行ごとに格納したリストを作成
    path_entext = Path(entext)
    with path_entext.open(mode='r') as f:
        l_entext = f.read().split('\n')

    df = pd.DataFrame(
        {"japanese_texts": l_jatext, "english_texts": l_entext})

    title_name = path_entext.stem
    # フォルダを作成
    title_folder = Path(fr'/workspace/data/title_test/{title_name}')
    title_folder.mkdir(parents=True, exist_ok=True)

    # 保存
    df.to_hdf(str(title_folder/fr'{title_name}.h5'), key='dataset')
