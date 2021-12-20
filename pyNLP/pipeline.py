from DataClean import dataClean
from FileReader import fileReader
from GetTopMentions import getTopMentionBySimilarity, getTopMentionByFrequency
import pandas as pd
if __name__ == '__main__':
    import sys

    dataPath = '/home/parallels/Desktop/Parallels Shared Folders/Home/Documents/wdps_assignment/final-wdps/data/'
    targetFile = 'bbc_omicron_1.csv'

    # read file
    print("-------Start Reading Data------"+"\n")
    file = fileReader(dataPath, targetFile)
    # read cheatsheet
    #  split 5 part
    # # cleandata
    print("-------Start Cleaning Data------"+"\n")
    df = dataClean(file,  URL=True, NUMBERS=True, LOWER=False,
                   EMOJI=True, CHARACTERS=True, STOPWORDS=False, EXPAND=True, LINES=True, STOPPOINT=True)
    # get top tweet sentence
    print("-------Start Getting Top Mentioned Triples------"+"\n")
    # countVector
    print("**Top Mentioned Triples: Frequencey**"+"\n")
    result1 = getTopMentionByFrequency(df["processed"], ngram=3)
    print(result1[:5])
    # similarity
    print("**Top Mentioned Triples: Similarity**"+"\n")
    result2 = getTopMentionBySimilarity(df['processed'].to_string(), ngram=3)
    print(result2)
    # -----------
    # cluster
