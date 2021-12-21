import numpy as np  # linear algebra
import re
from contractions import contractions_dict
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from string import digits
stop = set(stopwords.words('english'))
tokenizer = ToktokTokenizer()


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def expand_contractions(text, contraction_mapping=contractions_dict):
    contractions_pattern = re.compile('({})'.format(
        '|'.join(contraction_mapping.keys())), flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        try:
            expanded_contraction = contraction_mapping.get(match)\
                if contraction_mapping.get(match)\
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char+expanded_contraction[1:]
        except:
            pass
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_stopwords(text, is_lower_case=True, stopwords=stop):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def dataClean(file, URL, NUMBERS, LOWER, EMOJI, SPECIALCHARACTERS, STOPWORDS, EXPAND,LINES,STOPPOINT):
    # get the file with dataframe format
    df = file
    # caculate the repeats comments
    # if repeats from one user, the comment should be delete
    df['processed'] = df['text']
    # Remove URLS
    if URL:
        print("**Data Cleaning: Remove URL**"+"\n")
        df['processed'] = df['processed'].apply(
            lambda comment: re.sub(r"http\S+", "", comment))
    if LINES:
        print("**Data Cleaning: Remove Character Lines**"+"\n")
        df['processed'] = df['processed'].apply(lambda comment: re.sub("\\n"," ",comment))
        df['processed'] = df['processed'].apply(lambda comment: re.sub("(\w+)\: \“[^\”]*\”|(\w+)\:\“[^\”]*\”"," ", comment))
        df['processed'] = df['processed'].apply(lambda comment: re.sub("(\w+)\: \"[^\"]*\"|(\w+)\:\"[^\"]*\""," ", comment))
        df['processed'] = df['processed'].apply(lambda comment : re.sub(r"\“[^\”]*\”", " ", comment))
        df['processed'] = df['processed'].apply(lambda comment : re.sub(r"\"[^\"]*\"", " ", comment))

    if STOPPOINT:
        print("**Data Cleaning: Remove Time Break Point"+"\n")
        df['processed'] = df['processed'].apply(lambda comment : re.sub(r"(\d+)\:(\d+)"," ",comment))
    # expand words
    if EXPAND:
        print("**Data Cleaning: Expand Sentence**"+"\n")
        df['processed'] = df['processed'].apply(expand_contractions)
    # remove characters
    if SPECIALCHARACTERS:
        print("**Data Cleaning: Remove Special Character**"+"\n")
        df['processed'] = df['processed'].apply(remove_special_characters)
    # lower case
    if LOWER:
        print("**Data Cleaning: Lower the Character**"+"\n")
        df['processed'] = df['processed'].apply(
            lambda comment: comment.lower())
    if STOPWORDS:
        print("**Data Cleaning: Remove Stop Words**"+"\n")
        df['processed'] = df['processed'].apply(remove_stopwords)
    # remove emoji
    if EMOJI:
        print("**Data Cleaning: Remove Emojis**"+"\n")
        df["processed"] = df["processed"].apply(remove_emoji)
     # Remove numbers
    if NUMBERS:
        print("**Data Cleaning: Remove Numbers**"+"\n")
        df['processed'] = df['processed'].apply(
            lambda comment: re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", comment))
        #df['processed']=df['processed'].str.replace('d+','')

        df['processed'] = df['processed'].apply(
            lambda comment: re.sub(r"(\d+)", " ", comment))
        def keepAlpha(text):
            return text.translate(digits)
        df['processed'] =df['processed'].apply(keepAlpha)
    return df
