import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    format_search_result
)


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self):
        movie_dict = load_movies()
        for m in movie_dict:
            self.docmap[m['id']] = m
            self.__add_document(m['id'], f"{m['title']} {m['description']}")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_freq_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path):
            raise Exception("No index found")   
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_freq_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0
        for val in self.doc_lengths.values():
            total_length += val
        return total_length / len(self.doc_lengths)

    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise Exception("term must be a single token")
        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise Exception("term must be a single token")
        doc_count = len(self.docmap)
        df = len(self.index[tokens[0]])
        return math.log((doc_count + 1) / (df + 1))

    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise Exception("term must be a single token")
        doc_count = len(self.docmap)
        df = len(self.index[tokens[0]])
        return math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0: 
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1        
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
 
    def bm25(self, doc_id, term):
        bm25idf = self.get_bm25_idf(term)
        bm25tf = self.get_bm25_tf(doc_id, term)
        return bm25idf * bm25tf
    
    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=float(score),
                )
            )
        return results

def build_command():
    inv = InvertedIndex()
    inv.build()
    inv.save()

def preprocess_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def tokenize_text(text):
    tokens = preprocess_text(text).split()
    stop_words = load_stopwords()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    word_list = []
    for word in valid_tokens:
        if word not in stop_words:
            word_list.append(word)

    stemmer = PorterStemmer()
    stem_list = []
    for word in word_list:
        stem_list.append(stemmer.stem(word))

    return stem_list

def tf_from_doc(doc_id, term):
    inv = InvertedIndex()
    inv.load()
    return inv.get_tf(doc_id, term)

def term_idf(term):
    inv = InvertedIndex()
    inv.load()
    return inv.get_idf(term)

def tfidf_from_doc(doc_id, term):
    inv = InvertedIndex()
    inv.load()
    return inv.get_tfidf(doc_id, term)

def term_bm25idf(term):
    inv = InvertedIndex()
    inv.load()
    return inv.get_bm25_idf(term)

def bm25tf_from_doc(doc_id, term, k1=BM25_K1, b=BM25_B):
    inv = InvertedIndex()
    inv.load()
    return inv.get_bm25_tf(doc_id, term, k1, b)

def search_command(term, limit=DEFAULT_SEARCH_LIMIT):
    inv = InvertedIndex()
    inv.load()
    tokens = tokenize_text(term)
    results, result_ids = [], []
    for t in tokens:
        if len(results) >= limit:
            break
        result_ids.extend(inv.get_documents(t)[:limit])

    for id in result_ids:
        results.append(inv.docmap[id])
    
    return results

def bm25search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    inv = InvertedIndex()
    inv.load()
    return inv.bm25_search(query, limit)
