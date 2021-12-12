## NLP on YouTube Comments


In this project, I tried to extract comments from a list of videos returned from a search query on a youtube search engine. 

Using those comments, I did following analysis: 
- I created a word-cloud after preprocessing (including auto-translation to english) the text using [spaCy](https://spacy.io/).  
- I utilized the sentiment analysis pipeline provided by Hugging Face's library, [Transformers](https://github.com/huggingface/transformers), which is state-of-the-art for NLP.  
- I also used Named Entity Recognition (NER) pipeline from [Stanza](https://github.com/stanfordnlp/stanza), which is another one of the best NLP libraries developed by the Stanford NLP group. 

