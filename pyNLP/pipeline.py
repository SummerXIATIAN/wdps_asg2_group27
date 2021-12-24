from DataClean import dataClean
from FileReader import fileReader
from GetTopMentions import getTopMentionBySimilarity, getTopMentionByFrequency
import pandas as pd
import spacy
import collections
import nltk
import Utils
import DataClean
from nltk.corpus import stopwords
entity_dic = collections.defaultdict(int)
spacy_nlp = spacy.load("en_core_web_sm")

nltk.download('stopwords')
stop = set(stopwords.words('english'))
def compare(str1, str2):
    s1 = str1.split(" ")
    s2 = str2.split(" ")
    count = 1
    for a in s1:
        for b in s2:
            if a == b:
                count = count + 1
    if count >= (len(s1)/2+1):
        return True
    return False
def spacy_NER_process(text):
    s = spacy_nlp(text)
    result_dic = collections.defaultdict(int)
    for element in s.ents:
        if element.label_ in ['ORG', 'PRODUCT', 'PERSON']:
            result_dic[element.text.lower()] += 1
            entity_dic[element.text.lower()] += 1

    return result_dic.keys()

def top_word_reduction(df):
    #df = df.to_dict('split')['data']
    df["Flag"] = 0

    d1 = dict(zip(df['Word'], df['Frequency']))
    dflag = dict(zip(df['Word'], df['Flag']))
    d3 = dict(zip(df['Word'], df['Frequency']))

    for key in d1:
        for s in d3:
            if dflag[key] == 0 and dflag[s] == 0:
                if compare(key, s) and key != s:
                    if int(d1[key]) >= int(d1[s]):
                        d1[key] = int(d1[key])+int(d1[s])
                        dflag[s] = 1
                    else:
                        d1[s] = int(d1[key])+int(d1[s])
                        dflag[key] = 1

    d = []
    for key in d1:
        if dflag[key] == 0:
            d.append((key, d1[key]))
    return d



def Top_NER(targetFile):
    import sys

    dataPath = '/home/parallels/Desktop/Parallels Shared Folders/Home/Documents/wdps_assignment/final-wdps/data/'
    #dataPath = '../final-wdps/data/'
    versionDescriptionFile = 'version_description_015.txt'
    # read file
    print("-------Start Reading Data------"+"\n")
    result = fileReader(dataPath, targetFile, versionDescriptionFile)
    file = result[0]
    description = result[1].lower()
    main_focus_list = description.split("\n")
    # # cleandata
    print("-------Start Cleaning Data------"+"\n")
    df = dataClean(file,  URL=True, NUMBERS=True, LOWER=False,
                   EMOJI=True, SPECIALCHARACTERS=True, STOPWORDS=False, EXPAND=False, LINES=True, STOPPOINT=True)
    # spacy_NER and get the candidates
    df['spacy_NER_items'] = df['processed'].apply(spacy_NER_process)
    candidates = pd.DataFrame(
        entity_dic.items(), columns=['Word', 'Frequency'])
    candidates = pd.DataFrame(top_word_reduction(
        candidates), columns=['Word', 'Frequency'])
    candidates = candidates[candidates['Frequency'] > 50]
    Utils.export_bar_chart(candidates,targetFile)
    # combine the dictionary
    output =[]
    main_focus_list = candidates['Word']
    for focus in main_focus_list:
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"+"\n")

        def check(text):
            if focus in text:
                return True
            return False
        if Utils.link_to_fandom(focus):
            try:
                df_focus = df.loc[df['spacy_NER_items'].apply(check)]

                # get top tweet sentence
                print("-------Start Getting Top Mentioned Triples------"+"\n")
                # countVector
                print("**Top Mentioned Triples: Frequencey**"+"\n")

                df_focus['noun'] = df_focus['processed'].apply(
                    lambda comment: comment.lower()).apply(spacy_nlp)
                text = ''
                for i in df_focus['noun']:
                    for n in i.noun_chunks:
                        if n.text not in stop:
                            text = text+" "+n.text
                text = text+";"
                text = DataClean.remove_stopwords(text)
                top_bigrams = getTopMentionByFrequency(text.split(";"), 2)
                fr_top = pd.DataFrame(top_bigrams, columns=["Word", "Frequency"])
                reduced = top_word_reduction(fr_top)
                fr_top = pd.DataFrame(reduced, columns=['Word', 'Frequency'])
                
                fr_top=fr_top.sort_values("Frequency",ascending=False)[:5]
                print(fr_top.head())
                # similarity

                print("**Top Mentioned Triples: Similarity**"+"\n")
                sm_top = getTopMentionBySimilarity(
                    df_focus['processed'].to_string(), ngram=3)
                print(sm_top)
                link =  'https://genshin-impact.fandom.com/wiki/'+focus

                output.append((link,focus,fr_top['Word'].to_list(),sm_top))
                print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"+"\n"+"\n"+"\n")
            except:
                print(focus)
                continue
    output_df = pd.DataFrame(output,columns=["LINK","ENTITY","FR_TOP","SM_TOP"])
    output_df.to_csv("result"+targetFile)
    # -----------
    # cluster


if __name__ == '__main__':
    import sys
    list =['version_015_full.csv','version_016_full.csv','version_021_full.csv','version_022_full.csv','version_023_full.csv']
    for i in list:
        Top_NER(i)