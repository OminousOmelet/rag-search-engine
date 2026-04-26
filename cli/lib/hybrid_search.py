import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .utils import DEFAULT_QUERY_LIMIT, DEFAULT_ALPHA, RRF_K, RERANK_FACTOR, load_movies
from .enhance_query import enhance_query, rerank_func
from operator import itemgetter

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit=DEFAULT_QUERY_LIMIT):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha=DEFAULT_ALPHA, limit=DEFAULT_QUERY_LIMIT):
        bm25_results = self._bm25_search(query, limit*500)
        sem_results = self.semantic_search.search_chunks(query, limit*500)
        bm25_scores_only = list(map(lambda x: x['score'], bm25_results))
        sem_scores_only = list(map(lambda x: x['score'], sem_results))
        bm25_normed = normalize_list(bm25_scores_only)
        sem_normed = normalize_list(sem_scores_only)
        for i, doc in enumerate(bm25_results):
            doc['score'] = bm25_normed[i]
        for i, doc in enumerate(sem_results):
            doc['score'] = sem_normed[i]
        id_to_docs = {}
        for doc in bm25_results:
            id_to_docs[doc['id']] = {
                'title': doc['title'],
                'description': doc['document'],
                'bm25': doc['score']
            }
        for doc in sem_results:
            if doc['id']in id_to_docs:
                id_to_docs[doc['id']]['semantic'] = doc['score']
            else:
                id_to_docs[doc['id']] = {
                    'title': doc['title'],
                    'description': doc['document'],
                    'semantic': doc['score']
                }
        final_results = {}
        for id, doc in id_to_docs.items():
            if 'bm25' in doc and 'semantic' in doc:
                doc['hybrid'] = hybrid_score(doc['bm25'], doc['semantic'], alpha)
                final_results[id] = doc
        sorted_results = dict(
            sorted(final_results.items(), key=lambda x: x[1]['hybrid'], reverse=True)
        )
        return dict(list(sorted_results.items())[:limit])

    def rrf_search(self, query, k=RRF_K, limit=10): # magic number from instructor code, idk?
        bm25_results = self._bm25_search(query, limit*500)
        sem_results = self.semantic_search.search_chunks(query, limit*500)
        sorted_bm25 = sorted(bm25_results, key=itemgetter('score'), reverse=True)
        sorted_sem = sorted(sem_results, key=itemgetter('score'), reverse=True)
        ranked = {}
        for rank, doc in enumerate(sorted_bm25):
            ranked[doc['id']] = doc
            ranked[doc['id']]['bm25_rank'] = rank
            ranked[doc['id']]['rrf_score'] = 1.0 / float(k + rank)
        for rank, doc in enumerate(sorted_sem):
            if doc['id'] not in ranked:
                ranked[doc['id']] = doc
                ranked[doc['id']]['rrf_score'] = 0.0
            ranked[doc['id']]['sem_rank'] = rank
            ranked[doc['id']]['rrf_score'] += 1.0 / float(k + rank)
        return sorted(ranked.values(), key=lambda x: x['rrf_score'], reverse=True)[:limit]


def normalize_list(un_normed_list):
    if len(un_normed_list) == 0:
        return []
    norm_list = []
    min_val = min(un_normed_list)
    max_val = max(un_normed_list)
    if min_val == max_val:
        for i in range(len(un_normed_list)):
            norm_list.append(1.0)
    else:
        for arg in un_normed_list:
            norm_list.append((arg - min_val) / (max_val - min_val))
    return norm_list


def hybrid_score(bm25_score, semantic_score, DEFAULT_ALPHA=0.5):
    return DEFAULT_ALPHA * bm25_score + (1 - DEFAULT_ALPHA) * semantic_score


def weighted_search_command(query, DEFAULT_ALPHA, limit):
    hyb = HybridSearch(load_movies())

    results = hyb.weighted_search(query, DEFAULT_ALPHA, limit)
    for i, res in enumerate(results.values(), 1):
        print(f"{i}. {res['title']}")
        print(f"  Hybrid Score: {res['hybrid']:.3f}")
        print(f"  BM25: {res['bm25']:.3f}, Semantic: {res['semantic']:.3f}")
        print(f"  {res['description'][:100]}...")


def rrf_search_command(query, k, limit, enhancement=None, rerank_method=None):
    if rerank_method:
        limit *= RERANK_FACTOR

    final_query = ""
    if enhancement:
        enhanced_query = enhance_query(query, enhancement)
        print(f"\nEnhanced query ({enhancement}): '{query}' -> {enhanced_query}")
        final_query = enhanced_query
    else:
        final_query = query

    hyb = HybridSearch(load_movies())
    results = hyb.rrf_search(final_query, k, limit)
    if rerank_method:
        results = rerank_func(final_query, results, int(limit / RERANK_FACTOR), rerank_method)
    
    return results