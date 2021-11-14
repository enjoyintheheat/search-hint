from typing import List
from functools import lru_cache
from search_hint.common.settings import model_dir
from search_hint.modules.hint.correct import correction
import time
import spacy
import numpy as np
from annoy import AnnoyIndex
import pickle
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
from scipy import spatial
from dotenv import load_dotenv
import os
from sklearn.cluster import KMeans

# Пороговое значение для BERT-модели
THRESHOLD = 0.7
RESULTS_QTY = 10

# Глубина поиска. Чем больше - тем разнообразнее результаты, но медленнее
DEEP = 10


# %%
"""
### Loading models
"""

# %%

model = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')

nlp_ru = spacy.load("ru_core_news_lg", disable=['parser', 'ner'])

popular = AnnoyIndex(300, 'angular')
popular.load(os.path.join(model_dir, "query_popular_200.ann"))

other = AnnoyIndex(300, 'angular')
other.load(os.path.join(model_dir, "query_popular_other_200.ann"))

with open(os.path.join(model_dir, 'words_other.pkcls'), 'rb') as file:
    words_other = pickle.load(file)

with open(os.path.join(model_dir, 'words_popular.pkcls'), 'rb') as file:
    words_popular = pickle.load(file)


# %%
"""
### Main
"""

# %%


class TextProcessor:

    def __init__(self):
        pass

    def recognize(self, text: str) -> List[str]:
        if text == '':
            return []

        text = correction(text)

        v1 = model.encode(text)

        v = nlp_ru(text).vector
        v = np.array(v).astype(float).tolist()

        res_popular = popular.get_nns_by_vector(v, RESULTS_QTY*DEEP)
        res_other = other.get_nns_by_vector(v, RESULTS_QTY*DEEP)

        united = res_popular + res_other
        uni_dict = dict(words_other)
        uni_dict.update(words_popular)

        final = []
        for_bert = []

        for r in united:
            for_bert.append(uni_dict[r].lower())

        emb = model.encode(for_bert)

        distances = []
        for i, r in enumerate(united):
            dist = 1 - spatial.distance.cosine(emb[i], v1)
            if dist > THRESHOLD:
                final.append([dist, uni_dict[r].lower()])
                distances.append(emb[i])
                
        
        clf = KMeans(n_clusters=2, init='k-means++', n_init=100, random_state=42)
        labels = clf.fit_predict(distances)
        
        f = []
        previous_cluster = 0
        for index, sentence in enumerate(final):
            if index > 0:
                previous_cluster = labels[index - 1]
            cluster = labels[index]
            if previous_cluster != cluster:
                f.append([sentence[0], str(sentence[1])])
                
        final = np.array(sorted(f, key=lambda l:l[0], reverse=True))
        return final[:10, 1]


@lru_cache()
def text_processor_factory() -> TextProcessor:
    return TextProcessor()
