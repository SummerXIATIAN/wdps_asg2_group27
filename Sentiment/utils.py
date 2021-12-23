import re
import pandas as pd
from os import listdir


## read path and find csv files
def find_csv_filenames(path_to_dir=None, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    result = [filename for filename in filenames if filename.endswith( suffix )]
    return result

## convert 1,8k to 1800
def k_to_int(string):
    try:
        s = string.lower().replace(',',".")
        if "k" in s:
            value = float(s.split("k")[0])*1000
            return int(value)
        else:
            return int(s)
    except:
        return string
    
## remove special punctuation such as emoji
def removePunctuation(line):
    line = str(line)
    line = line.replace("\n"," ").replace("\r"," ")
    if line.strip() == '':
        return ''
    line = re.sub("[^\u4e00-\u9fa5^\s\.\!\:\-\@\#\$\(\)\_\,\;\?^a-z^A-Z^0-9]","",line)
    return line
    
## label for textblob senti score
def scoreRange_blob(score):
    s = float(score)
    if s <= -0.5:
        return -2
    elif s <= -0.1:
        return -1
    if s >= 0.5:
        return 2
    elif s >= 0.1:
        return 1
    else:
        return 0

####### utility functions for bili (CN) comments
def txt_to_df(file):
    index = []
    comments = []
    with open(file, "r") as f:
        data = f.read()

    for i in data.split("\n"+"-"*10+"\n"):
        pos = i.find("、")
        index.append(i[:pos])
        comments.append(i[pos+1:])
    df = pd.DataFrame({'index':index,"content":comments})
    return df

## remove non-Chinese word
def isZN(line):
    line = str(line)
    if line.strip() == '':
        return False
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    line = rule.sub('', line)
    if line == '':
        return False
    else:
        return True

## Chinese word data cleaning
def removePunctuation_CN(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(
        u"[^ a-zA-Z0-9\u4E00-\u9FA5\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]")
    line = rule.sub('', line)
    return line

## label for snownlp senti score
def scoreRange_snow(score):
    s = float(score)
    if s <= 0.2:
        return -2
    elif s <= 0.4:
        return -1
    if s >= 0.8:
        return 2
    elif s >= 0.6:
        return 1
    else:
        return 0