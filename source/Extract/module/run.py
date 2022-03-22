import pandas as pd
from module.tf_idf import Converter
from module.extract import Extract
from module.image_create import Image_create

def Main(title, num):
    num = num
    title = title
    converter = Converter(title)
    idf, lists = converter.main()
    method = ["Line drawing","oil painting", "Monochrome painting", "Watercolor painting", 
          "ink painting", "prints", "illustration", "tempera", "Oshie", "woodblock print"]
    extract = Extract(title, lists)
    df = extract.main()
    text_list = df['english_texts'].tolist()
    
    for i, text in enumerate(text_list):
        directory = "{}_{}".format(title, i)
        image = Image_create(text, num, directory, method, title)
        image.Create()