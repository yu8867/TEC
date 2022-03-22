import pandas as pd

class Extract:
    def __init__(self, title, tfidf):
        """クラスの初期化
        Args:
            title : 作品のタイトル
            tfidf : TF-IDFから抽出した語句
        """
        
        hdf_path = '/workspace/data/title/'
        file_name = '{}_predicted.h5'.format(title)
        self.df = pd.read_hdf(hdf_path+file_name)
        self.lists = tfidf
        
    def Drop(self):
        """ データのクリーニング """
        kakko = ["「", "」"]
        short_index = [i for i,x in enumerate(self.df['japanese_texts']) if len(x) <= 15]
        kakko_index = [i for i,x in enumerate(self.df['japanese_texts']) if x[0] in kakko]

        drop_index = kakko_index + short_index
        drop_index = list(set(drop_index))

        self.df = self.df.drop(drop_index)
        self.df = self.df.reset_index(drop=True)


        self.df = self.df.query('contexsts_label=="1" & picture_label=="1"')
        self.df = self.df.reset_index(drop=True)
        
    def Extract_index(self):
        """ データの抽出 """
        index = []
        for i in range(len(self.df)):
            for item in self.lists:
                # 入ってる
                if item in self.df['japanese_texts'][i]:
                    index.append(i)
                    break
                    
        print(index)

        tfidf_df = self.df.iloc[index]
        tfidf_df = tfidf_df.reset_index(drop=True)
        return tfidf_df
        
    def main(self):
        """ メイン """
        self.Drop()
        df = self.Extract_index()
        return df
