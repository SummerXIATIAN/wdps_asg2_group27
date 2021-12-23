import pandas as pd
import smoothnlp as slp
import matplotlib.pyplot as plt
from snownlp import SnowNLP
from smoothnlp.algorithm.phrase import extract_phrase
from utils import *

senti_ref = {2:"Postive", 1:"Semi-Positive", 0:"Normal", -1:"Semi-Negative", -2:"Negative"}
path = "./data/bili/"

## read data from txt file
def readData(path):
    df = txt_to_df(path)
    df = df[df.content.apply(lambda x: len(str(x)) < 500)]
    df['content'] = df.content.apply(removePunctuation_CN)
    df = df[df.content.map(isZN)]
    df = df.dropna(subset=['content'])
    df.drop_duplicates(subset="content", inplace=True)
    df = df.reset_index(drop=True)
    return df

## Calculate senti and label them
def sentiScore_CN(df):
    df["score"] = df.content.apply(lambda x: SnowNLP(x).sentiments)
    df['label'] = df.score.apply(scoreRange_snow)
    return df

## extract keywords from content
def getKeywords(df):
    corpus = list(df.content)
    wordlist = extract_phrase(corpus, top_k=20, min_n=2, max_n=4, min_freq=2)
    return wordlist


## draw pie plot for senti distribution
def pie_plot(dataframe=None,save=False,ref=[]):
    v = dataframe.label.value_counts().sort_index(ascending=False)
    quants   = list(v)
    labels   = v.index.tolist()
    labels = list(map(senti_ref.get, labels)) if bool(ref) == True else labels

    # Pie Plot
    plt.figure(1, figsize=(7,7))
    expl = [0.02,0.02,0.1,0.02,0.02] # make the piece explode a bit
    colors  = ["#1BFF00","#7DFF44","#FFEC00","#FF5C30","#FF2E00"] # Colors used. Recycle if not enough.
    plt.pie(quants, explode=expl, colors=colors, labels=labels, autopct='%1.1f%%', \
        textprops={'fontsize': 14}, pctdistance=0.7, shadow=True) # autopct: format of "percent" string
    plt.title('Sentiment Distribution', bbox={'facecolor':'0.9', 'pad':5})
    if save: # save plot
        plt.savefig(f'{str(save)}.png')
    else:
        plt.show()
    plt.close()
    return quants,labels


if __name__ == '__main__':
    import sys
    try:
        _, PATH = sys.argv
    except Exception as e:
        print('No argument for data path')
        PATH = path

    files = find_csv_filenames(PATH,suffix=".txt")
    files.sort()
    history = [];kw = []
    for file in files:
        filename = file
        filepath = str(PATH + filename)
        print(f'In process {filename}..')

        data = readData(filepath)
        data = sentiScore_CN(data)
        keywords = getKeywords(data)
        quants,_ = pie_plot(dataframe=data,save=filename[:-4],ref=senti_ref)
        history.append(data.score.mean())
        kw.append(keywords)

    ## plot historial average senti score (trend)
    x = [f[:f.find(".txt")] for f in files]
    y = [round(s,4) for s in history]
    plt.figure(1, figsize=(14,7))
    plt.plot(x,y,linestyle = '-',linewidth = 3,color = 'b',markersize = 5,markerfacecolor='g')
    plt.ylim(0.5,0.7)
    plt.xlabel('PV name')
    plt.ylabel('Average senti score')
    plt.title('Sentiment Score Trend')
    plt.savefig('bili_trend.png')
    plt.close()
    print(kw)