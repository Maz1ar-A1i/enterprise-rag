import sys
sys.path.insert(0, r'd:\Project 3')

errors = []

tests = [
    ('faiss',               'import faiss; v=getattr(faiss,"__version__","ok"); print(f"  faiss {v}")'),
    ('sentence_transformers','import sentence_transformers as st; print(f"  sentence_transformers {st.__version__}")'),
    ('rank_bm25',           'from rank_bm25 import BM25Okapi; print("  rank_bm25 OK")'),
    ('groq',                'import groq; print(f"  groq {groq.__version__}")'),
    ('tiktoken',            'import tiktoken; print("  tiktoken OK")'),
    ('datasketch',          'from datasketch import MinHash, MinHashLSH; print("  datasketch OK")'),
    ('bs4',                 'from bs4 import BeautifulSoup; print("  bs4 OK")'),
    ('markdownify',         'from markdownify import markdownify; print("  markdownify OK")'),
    ('tenacity',            'from tenacity import retry; print("  tenacity OK")'),
    ('psutil',              'import psutil; print("  psutil OK")'),
    ('sklearn',             'from sklearn.metrics import ndcg_score; print("  sklearn OK")'),
    ('matplotlib',          'import matplotlib; print(f"  matplotlib {matplotlib.__version__}")'),
    ('tabulate',            'from tabulate import tabulate; print("  tabulate OK")'),
    ('numpy',               'import numpy as np; print(f"  numpy {np.__version__}")'),
    ('yaml',                'import yaml; print("  yaml OK")'),
    ('dotenv',              'from dotenv import load_dotenv; print("  dotenv OK")'),
    ('cleaner',             'from src.corpus.cleaner import clean_document; print("  cleaner OK")'),
    ('chunker',             'from src.corpus.chunker import chunk_document, Chunk; print("  chunker OK")'),
    ('deduplicator',        'from src.corpus.deduplicator import deduplicate; print("  deduplicator OK")'),
    ('bm25_retriever',      'from src.retrieval.bm25_retriever import BM25Retriever; print("  bm25_retriever OK")'),
    ('dense_retriever',     'from src.retrieval.dense_retriever import DenseRetriever; print("  dense_retriever OK")'),
    ('hybrid_retriever',    'from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult; print("  hybrid_retriever OK")'),
    ('reranker',            'from src.retrieval.reranker import Reranker; print("  reranker OK")'),
    ('groq_client',         'from src.generation.groq_client import GroqClient; print("  groq_client OK")'),
    ('prompt_builder',      'from src.generation.prompt_builder import build_prompt, is_refusal, extract_citations; print("  prompt_builder OK")'),
    ('naive_rag',           'from src.pipeline.naive_rag import NaiveRAGPipeline, RAGResult; print("  naive_rag OK")'),
    ('metrics',             'from src.evaluation.metrics import recall_at_k, mean_reciprocal_rank, ndcg_at_k; print("  metrics OK")'),
    ('benchmark_runner',    'from src.evaluation.benchmark_runner import BenchmarkRunner; print("  benchmark_runner OK")'),
    ('report_generator',    'from src.evaluation.report_generator import ReportGenerator; print("  report_generator OK")'),
]

print("Testing all imports...")
for name, code in tests:
    try:
        exec(code)
    except Exception as e:
        errors.append(f"  FAIL: {name} -> {e}")
        print(f"  FAIL: {name} -> {e}")

print()
if errors:
    print(f"FAILED: {len(errors)} imports failed")
    for e in errors:
        print(e)
else:
    print("ALL IMPORTS PASSED - System is ready!")
