from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import numpy as np
pipeline = 'distilbert-base-nli-mean-tokens'
model = SentenceTransformer(pipeline)

def getTopMentionByFrequency(corpus, ngram=1, n=None):
    n_gram_range = (ngram, ngram)
    vec = CountVectorizer(ngram_range=n_gram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def getTopMentionBySimilarity(doc, ngram=1, stopWords="english"):
    n_gram_range = (ngram, ngram)
    count = CountVectorizer(ngram_range=n_gram_range,
                            stop_words=stopWords).fit([doc])
    candidates = count.get_feature_names()
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)

    def max_sum_sim(doc_embedding, word_embeddings, words, top_n, nr_candidates):
        # Calculate distances and extract keywords
        distances = cosine_similarity(doc_embedding, word_embeddings)
        distances_candidates = cosine_similarity(
            word_embeddings, word_embeddings)

        # Get top n words as candidates based on cosine  similarity
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [words[index] for index in words_idx]
        distances_candidates = distances_candidates[np.ix_(
            words_idx, words_idx)]

        # Calculate the combination of words that are the least similar to each other
        min_sim = np.inf
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum([distances_candidates[i][j]
                      for i in combination for j in combination])
            if sim < min_sim:
                candidate = combination
                min_sim = sim
        return [words_vals[idx] for idx in candidate]
    return max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10)
