from DataClean import dataClean
from FileReader import fileReader
from GetTopMentions import getTopMentionBySimilarity, getTopMentionByFrequency
import pandas as pd
import spacy
import collections
entity_dic = collections.defaultdict(int)
spacy_nlp = spacy.load("en_core_web_sm")


def spacy_NER_process(text):
    s = spacy_nlp(text)
    result_dic = collections.defaultdict(int)
    for element in s.ents:
        if element.label_ in ['ORG', 'PRODUCT', 'PERSON']:
            result_dic[element.text.lower()] += 1
            entity_dic[element.text.lower()] += 1

    return result_dic.keys()


if __name__ == '__main__':
    import sys

    dataPath = '/home/parallels/Desktop/Parallels Shared Folders/Home/Documents/wdps_assignment/final-wdps/data/'
    targetFile = 'version_015_full.csv'
    versionDescriptionFile = 'version_description_015.txt'
    # read file
    print("-------Start Reading Data------"+"\n")
    result = fileReader(dataPath, targetFile, versionDescriptionFile)
    file = result[0]
    description = result[1].lower()
    main_focus_list = description.split("\n")
    # read cheatsheet
    #  split 5 part
    # # cleandata
    print("-------Start Cleaning Data------"+"\n")
    df = dataClean(file,  URL=True, NUMBERS=True, LOWER=False,
                   EMOJI=True, SPECIALCHARACTERS=True, STOPWORDS=False, EXPAND=False, LINES=True, STOPPOINT=True)
    # spacy_NER
    df['spacy_NER_items'] = df['processed'].apply(spacy_NER_process)
    for focus in main_focus_list:
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"+"\n")

        def check(text):
            if focus in text:
                return True
            return False
        try:
            df_focus = df.loc[df['spacy_NER_items'].apply(check)]

            # get top tweet sentence
            print("-------Start Getting Top Mentioned Triples------"+"\n")
            # countVector
            print("**Top Mentioned Triples: Frequencey**"+"\n")
            result1 = getTopMentionByFrequency(df_focus["processed"], ngram=2)
            print(result1[:20])
            # similarity

            print("**Top Mentioned Triples: Similarity**"+"\n")
            result2 = getTopMentionBySimilarity(
                df_focus['processed'].to_string(), ngram=2)
            print(result2)
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"+"\n"+"\n"+"\n")
        except:
            print(focus)
            continue
    
    # -----------
    # cluster
