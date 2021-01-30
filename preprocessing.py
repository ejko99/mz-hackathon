import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import csv
import re


# 어절(띄어쓰기) 기준 tokenizing
def tokenizing_text(texts):
    corpus = []
    for s in texts:
        result = re.split(' ',str(s))
        corpus.append(result)
    return corpus

# sentence summation
def str_sum(text):
    temp = list()
    for s in text:
        temp.append(' '.join(s))
    return temp

def pre_processing(text_data):

    data = pd.read_csv(text_data ,header = None, names = ['text'],encoding='cp949')

    # text col
    text = data['text']
    

    x_test = text
 
    #write in tsv
    with open('test.tsv', 'wt', newline='', encoding='utf-8-sig') as f:
        print('Write text data to {} ...'.format('test.tsv'))
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x_test))

# if __name__ == "__main__":

#     pre_processing('dev_nolabel.txt')