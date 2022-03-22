import MeCab
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import Counter
import warnings
from glob import glob
from pathlib import Path
import requests
warnings.simplefilter('ignore', category=RuntimeWarning)

class Converter:
    def __init__(self, title):
        """クラスの初期化
        Args:
            title : 作品のタイトル
        """
        
        self.title = title
        url = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
        r = requests.get(url)
        tmp = r.text.split('\r\n')
        stopwords_jp = []
        for i in range(len(tmp)):
            if len(tmp[i]) < 1:
                continue
            stopwords_jp.append(tmp[i])
        self.stopwords_jp = stopwords_jp

    def convert(self):
        """ 日本語テキスト """
        path_jp = "/workspace/data/inputs/{}.txt".format(self.title)
        texts_jp = glob(path_jp)
        input_txt_jp = []

        for txt in texts_jp:
            p_txt = Path(txt)
            title = p_txt.name
            with p_txt.open(mode='r') as f:
                input_txt_jp = f.read().split('\n')

        input_txt_jp = ''.join(input_txt_jp)
        
        text = re.sub(r'《.+?》', '', input_txt_jp)
        text = re.sub(r'［＃.+?］', '', text)
        text = re.sub(r'｜', '', text)
        text = re.sub(r'\r\n', '', text)
        text = re.sub(r'\u3000', '', text)  
        text = re.sub(r'「', '', text) 
        text = re.sub(r'」', '', text)
        text = re.sub(r'、', '', text)
        text = re.sub(r'。', '', text)   

        return text

    def wakati(self, string):
        """ わかち """
        select_conditions = ['固有名詞','名詞','人名', '副詞可能','形状詞可能']
        tagger = MeCab.Tagger('')
        node = tagger.parseToNode(string)
        terms = []

        while node:
            term = node.surface
            pos = node.feature.split(',')[0]
            pos_1 = node.feature.split(',')[1]
            pos_2 = node.feature.split(',')[2]
            
            if pos in select_conditions[0]:
                if pos_1 not in select_conditions[1]:
                    if pos_2 not in select_conditions[1:]:
                        if term not in self.stopwords_jp:
#                             print(term)
#                             print(node.feature,"\n")
                            terms.append(term)
            node = node.next
            
        text_result = ' '.join(terms)
        return text_result

    def Vectorizer(self, docs):
        """ 単語ベクトルの類似 """
        vectorizer = TfidfVectorizer(smooth_idf=False)
        X = vectorizer.fit_transform(docs)
        values = X.toarray()
        feature_names = vectorizer.get_feature_names()
        df = pd.DataFrame(values, columns = feature_names,
                      index=[self.title])
        return df

    def Count(self, docs):
        """ 語句のカウント """
        docs_ = docs[0].split(" ")
        # 出現回数の取得
        cnt = Counter(docs_)
#         display(cnt.most_common(20)) #上位10

    def visual(self, df):
        """ データセットの可視化 """
        df_0 = df[0:1].T
        df_0 = df_0.sort_values(by=self.title, ascending=False)
#         display(df_0[:20])
        return df_0

    def main(self):
        """ メイン """
        docs = []
        lists = []
        string = self.convert()

        text = self.wakati(string)
        docs.append(text)

        df = self.Vectorizer(docs)
        df_0 = self.visual(df)

        self.Count(docs)
        
        for i in range(5):
            lists.append(df_0.index[i])
        return df_0, lists
