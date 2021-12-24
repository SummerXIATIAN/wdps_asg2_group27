import pandas as pd
import re
from string import digits

from matplotlib import pyplot as plt
from textblob import TextBlob
import glob


def save_english(text_list):
    for i in range(len(text_list)):
        text_list[i] = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", " ", text_list[i])
        table = str.maketrans('', '', digits)
        text_list[i] = text_list[i].translate(table)


csv_list = glob.glob('Data/*.csv')
for i in csv_list:
    df = pd.read_csv(i)
    text_list = df['text'].to_list()
    save_english(text_list)

    score_list = []
    for text in text_list:
        s = TextBlob(text)
        score = s.sentiment
        score_list.append(score)
    df['score'] = score_list
    positive = 0
    negative = 0
    for j in score_list:
        if j.polarity>=0:
            positive +=1
        else:
            negative +=1

    file_name = i.split('\\')[1].split('.')[0]+'_result.csv'
    df.to_csv('TB_result/'+file_name)

    plt_name = i.split('\\')[1].split('.')[0] + '_plt.jpg'
    labels = ['positive', 'negative']
    sizes = [positive, negative]
    patches, text1, text2 = plt.pie(sizes, labels=labels, autopct='%1.2f%%')
    plt.savefig('TB_result/' + plt_name)
