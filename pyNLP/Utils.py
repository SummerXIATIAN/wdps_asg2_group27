import requests
import spacy
import collections
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import seaborn as sns
sns.set_style('darkgrid')
spacy_nlp = spacy.load("en_core_web_sm")

nltk.download('stopwords')
stop = set(stopwords.words('english'))
def export_bar_chart(df,file):
    sns.barplot(df['Frequency'],df['Word'])
    plt.savefig(file+'.png')

def link_to_fandom (mention):
    newstring ='' 
    upcase = True
    for a in mention: 
        if upcase == True:
            newstring+=(a.upper()) 
            upcase = False
        elif (a.isspace()) == True:
            newstring+='_'
            upcase = True
        else:
            newstring+=a
#print(newstring)     

    url = 'https://genshin-impact.fandom.com/wiki/'+newstring
    r = requests.get(url)
    if r.status_code != 404:
        return True
    return False






