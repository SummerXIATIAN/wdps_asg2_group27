from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

senti_ref = {2:"Postive", 1:"Semi-Positive", 0:"Normal", -1:"Semi-Negative", -2:"Negative"}
path = "./data/"

## Read csv file, clean and reorganize
def readData(filepath):
    df = pd.read_csv(filepath,error_bad_lines=False,lineterminator='\n')
    df['votes'] = df.votes.apply(k_to_int)
    df["content"] = df.text.apply(removePunctuation)
    df['length'] = df.content.apply(lambda x: len(x))
    df = df[~df['content'].str.contains('“|"|:')] # remove quote comments
    df = df[(df.length<1024)&(df.length>6)]
    df = df.sort_values("votes",ascending=False)
    df = df.dropna(subset=['content'])
    df.drop_duplicates(subset="content", inplace=True)
    df = df.reset_index(drop=True)
    df = df[['text','votes','content','length']]
    return df

## Calculate senti and subjectivity score
def sentiScore(df):
    df["score"] = df.content.apply(lambda x: TextBlob(x).sentiment.polarity)
    df["subjectivity"] = df.content.apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['label'] = df.score.apply(scoreRange_blob) # create label via score
    df = df[~(df.subjectivity==0)] # remove no subject comments
    df = df.reset_index(drop=True)
    return df

## select comments that include keywords
def filterKeyWords(keyword,dataframe):
    df = dataframe.copy()
    df = df[df['content'].str.contains(keyword,case=False)]
    df = df.reset_index(drop=True)
    return df

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

    files = find_csv_filenames(PATH,suffix="l.csv")
    files.sort()
    history = []
    for file in files:
        filename = file
        filepath = str(PATH + filename)
        print(f'In process {filename}..')

        data = readData(filepath)
        data = sentiScore(data)
        # print(data.head())
        quants,_ = pie_plot(dataframe=data,save=filename[:-4],ref=senti_ref)
        history.append(data.score.mean())

    ## plot historial average senti score (trend)
    x = [f[:f.find(".csv")] for f in files]
    y = [round(s,4) for s in history]
    plt.figure(1, figsize=(14,7))
    plt.plot(x,y,linestyle = '-',linewidth = 3,color = 'b',markersize = 5,markerfacecolor='g')
    plt.ylim(0.1,0.24)
    plt.xlabel('PV name')
    plt.ylabel('Average senti score')
    plt.title('Sentiment Score Trend')
    plt.savefig('trend.png')
    plt.close()