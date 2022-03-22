import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.max_colwidth", 300)
import numpy as np
from tqdm import tqdm
import random
import os
import re
import requests
import warnings
from glob import glob
from pathlib import Path
from PIL import Image
import MeCab
import nltk
import string
from nltk.tag import pos_tag
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('words')

class convert_jp:
    def __init__(self, stopwords_jp):
        """クラスの初期化
        Args:
            stopwords_jp : ストップワード
        """
        
        self.words = []
        self.entity_jp = []
        self.stopwords_jp = stopwords_jp
        self.text = "a"
        
    def convert(self, text):
        """ 文章のクリーニング """
        text = re.sub(r'《.+?》', '', text)
        text = re.sub(r'［＃.+?］', '', text)
        text = re.sub(r'｜', '', text)
        text = re.sub(r'\r\n', '', text)
        text = re.sub(r'\u3000', '', text)  
        text = re.sub(r'「', '', text) 
        text = re.sub(r'」', '', text)
        text = re.sub(r'、', '', text)
        self.text = re.sub(r'。', '', text)  
        
    def wakati(self):
        """ 特定の詞を抽出"""
        select_conditions = ['名詞']
        tagger = MeCab.Tagger('')
        node = tagger.parseToNode(self.text)
        terms = []

        while node:
            term = node.surface
            pos = node.feature.split(',')[0]
            if pos in select_conditions:
                terms.append(term)
            node = node.next
        self.words = terms
        
        self.entity_jp = [word for word in self.words if word not in self.stopwords_jp]
        self.entity_jp = '/'.join(self.entity_jp)
    
    def main(self, text):
        """ メイン """
        self.convert(text)
        self.wakati()
        return self.entity_jp
    
class convert_en:
    def __init__(self, stopwords_en):
        """クラスの初期化
        Args:
            stopwords_en : 英語のストップワード
        """
        
        self.stopwords_en = stopwords_en
        self.entity_en = []
        self.text = "a"
    
    def convert(text):
        """ 文章のクリーニング """
        text = text.replace(",","")
        self.text = text.replace(".","")
    
    def nlpk(self):
        """ 特定の詞を抽出"""
        pos = nltk.word_tokenize(self.text)
        pos_im = nltk.pos_tag(pos)
        
        for i in range(len(pos_im)):
            if pos_im[i][1] == "NNP" or pos_im[i][1] == "NN":
                words.append(pos_im[i][0])
        
        entity_en = [word for word in words if word not in self.stopwords_en]
        self.entity_en = "/".join(entity_en)
        return self.entity_en
        
    def main(self, text):
        """ メイン """
        self.convert(text)
        self.nlpk()
        return self.entity_en
    
def print_hl(text, keyword, color="yellow"):
    """ 名詞・固有名詞のハイライト """
    color_dic = {'yellow': '\033[43m', 'red': '\033[31m', 'blue': '\033[34m', 'end': '\033[0m'}
    for kw in keyword:
        bef = kw
        aft = color_dic[color] + kw + color_dic["end"]
        text = re.sub(bef, aft, text)
    print(text)
    

class dataset: 
    def __init__(self, title, num):
        """クラスの初期化
        Args:
            title : 作品
            num : 欲しいデータ数
        """
        
        self.title = title
        self.num = num
        
        self.path_en = "/workspace/data/outputs/{}.txt".format(title)
        self.path_jp = "/workspace/data/inputs/{}.txt".format(title)
        self.image = "/workspace/data/img/" + title
        # /workspace/data/hdf5_suzuki/
        self.hdf5 = "/workspace/data/hdf5_kobayashi/"
        
        self.texts_en = glob(self.path_en)
        self.texts_jp = glob(self.path_jp)
        self.files = os.listdir(self.image)
        
        # 特徴量
        self.input_txt_en = []
        self.input_txt_jp = []
        self.image_name = []
        self.path_name = []
        
        # アノテーションのラベル
        self.contexts_short = []
        self.image_label = []
        self.entity_jp_label = []
        self.entity_en_label  = []
        
        self.df = pd.DataFrame()
        
        # 固有表現
        self.entity_en = []
        self.entity_jp = []
        
        # stop_words
        url = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
        r = requests.get(url)
        tmp = r.text.split('\r\n')
        stopwords_jp = []
        for i in range(len(tmp)):
            if len(tmp[i]) < 1:
                continue
            stopwords_jp.append(tmp[i])
            
        # jp_stopwords
        self.stopwords_jp = stopwords_jp
        self.convert_jp = convert_jp(self.stopwords_jp)
        
        # en_stopwords
        self.stopwords_en = stopwords.words('english')
        self.convert_en = convert_jp(self.stopwords_en)
        
        # random_sampling
        self.random_list = []
        self.index = []
        self.use_index = []
        
    def Set(self):
        """ データの加工・特徴量の作成 """
        """ 英語テキスト """
        for txt in self.texts_en:
            p_txt = Path(txt)
            title = p_txt.name
            with p_txt.open(mode='r') as f:
                self.input_txt_en = f.read().split('\n')
        
        """ 日本語テキスト """
        for txt in self.texts_jp:
            p_txt = Path(txt)
            title = p_txt.name
            with p_txt.open(mode='r') as f:
                self.input_txt_jp = f.read().split('\n')
        
        """ 画像の名前 """
        for i in range(len(self.input_txt_jp)):
            filename = "{}.png".format(i)
            self.image_name.append(filename)
            
        """ パス """
        for _ in range(len(self.input_txt_jp)):
            path_png = "/workspace/data/img/{}.png".format(_)
            self.path_name.append(path_png)
            
        """ 重要語_日本語 """
        for i in range(len(self.input_txt_jp)):
            words_en = self.convert_en.main(self.input_txt_en[i])
            words_jp = self.convert_jp.main(self.input_txt_jp[i])
            self.entity_en.append(words_en)
            self.entity_jp.append(words_jp)
            
        
        """ データセット作成 """
        self.df = pd.DataFrame({"japanese_texts":self.input_txt_jp, "English_texts":self.input_txt_en, 
                                "picture_name":self.image_name,"path":self.path_name,
                                "entity_jp":self.entity_jp, "entity_en":self.entity_en})
        # data/title/df_{title}
        
        """ 文字数の短い・「」で始まる文章の削除 """
        kakko = ["「", "」"]
        drop_kakko_index = [i for i,x in enumerate(self.input_txt_jp) if x[0] in kakko]
        drop_short_index = [i for i,x in enumerate(self.input_txt_jp) if len(x) <= 15]
        
        drop_index = drop_kakko_index + drop_short_index
        drop_index = list(set(drop_index))
        self.use_index = [i for i in range(len(self.df)) if i not in drop_index]

        self.Calculate()
        
        """ 使えるデータのインデックスを所得 """
        self.random_list = random.sample(self.use_index, k=self.num)
        
    def Create_dataset(self):
        """ concat """
        self.df = self.df.iloc[self.index]
        self.df["contexsts_label"] = self.contexts_short
        self.df["picture_label"] = self.image_label
        
        display(self.df)

    def show_image(self, index):
        """ 画像の表示 """
        path = self.image + "/{}.png".format(index)
        im = Image.open(path)
        im_list = np.asarray(im)
        plt.imshow(im_list)
        plt.show()
    
    def Calculate(self):
        """ データの数を確認 """
        print("-"*108)
        print("もしデータの数が合わなかったら、合うように整形してください")
        print("   ・英語の文章  ",len(self.input_txt_en))
        print("   ・日本語の文章",len(self.input_txt_jp))  
        print("   ・画像数     ",len(self.image_name))
        print("   ・パス       ",len(self.path_name))
        print("   ・使用できるデータ数",len(self.use_index))
        print("-"*108,"\n")
        
    def Label(self):
        """ アノテーション """
        for i in tqdm(self.random_list):
            print(i,"番目")
            
            """ 重要語のハイライト """
            print_hl(self.df["japanese_texts"][i], self.df["entity_jp"][i].split("/"), color="red")
            print("\n")
#             print_hl(self.df["English_texts"][i], self.df["entity_en"][i].split("/"), color="red")
#             print("\n")
            
            """ テキスト連想 """
            while (True):
                ans = input("0:連想できない       1:連想できる")
                if ans=="1" or ans=="0":
                    break
            self.contexts_short.append(int(ans))
            

            """ 画像の正確 """
            self.show_image(i)
            print("画像は文章を表せていますか？\n")
            while (True):
                ans = input("0:No           1:Yes")
                if ans=="1" or ans=="0":
                    break
            self.image_label.append(int(ans))
            self.index.append(i)
            print("\n")
    
    def Save(self):
        """ datasetの保存 """
        hdf_folder = Path(self.hdf5)
        self.df.to_hdf(self.hdf5+"{}.h5".format(self.title),key="datasets")
        # ここのpathを/workspace/data/csv/
        self.df.to_csv("/workspace/data/csv_kobayashi/{}.csv".format(self.title))

    def main(self):
        """ メイン """
        self.Set()
        self.Label()
        self.Create_dataset()
        self.Save()