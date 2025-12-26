# ablation_runner_graph_sentences_MATCH_INITIAL_HYBRID_EXTRACTION.py
#
# This script is an ablation runner that matches the INITIAL approach as closely as possible,
# with IMPROVED HYBRID ANSWER EXTRACTION using LLM + semantic similarity + regex cascade.
#
# Clean-factorial fixes included:
# - Decouple rerank from retrieval mode (rerank can be ON/OFF for BM25/VECTOR/ENSEMBLE)
# - Graph OFF truly skips graph extraction/merging/sentence generation
# - Adds CLEAN__R=...__RR=...__G=... ablations (optional)
# - Adds retrieval_mode/rerank_on/graph_on columns to output rows (to avoid analysis confusion)
#
# Run:
#   python ablation_runner_graph_sentences_MATCH_INITIAL_HYBRID_EXTRACTION.py \
#     --input_csv in.csv --output_csv out.csv \
#     --glirel_model_path ./Drugprot_REL_model \
#     --spacy_model_path ./NER_Model/model-best \
#     --chroma_dir ./vectorstore_antiviral_chunk_size_600 \
#     --bm25_dir bm25_antivir
#
# Eval:
#   python ablation_runner_graph_sentences_MATCH_INITIAL_HYBRID_EXTRACTION.py --eval_stats --output_csv out.csv
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
import logging

# Silence transformer/sentence-transformer progress bars
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import re
import gc
import json
import argparse
import random
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import spacy
import networkx as nx

from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

from glirel import GLiREL
from glirel.modules.utils import constrain_relations_by_entity_type

from sentence_transformers import SentenceTransformer

from langchain_community.retrievers import BM25SRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama


# -----------------------
# Logging / warnings (match INITIAL: keep it minimal)
# -----------------------
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*INFO:.*")
warnings.filterwarnings("ignore")


# -----------------------
# Seeding (DO NOT force deterministic kernels; INITIAL didn't)
# -----------------------
def set_global_seed(seed: int):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------
# Constants
# -----------------------
OPTION_LETTER_TO_COP = {"a": 1, "b": 2, "c": 3, "d": 4}

FULL_SYSTEM = "FULL_SYSTEM"
NO_GRAPH_AUG = "NO_GRAPH_AUGMENTATION"
DENSE_ONLY = "DENSE_ONLY_RETRIEVAL"
BM25_ONLY = "BM25_ONLY_RETRIEVAL"
NO_MERGE = "NO_ENTITY_MERGING"
EDGES_OFF = "EDGES_OFF"
K_LOW = "TOPK_K_LOW"
K_DEFAULT = "TOPK_K_DEFAULT"

ABLATIONS = [FULL_SYSTEM, NO_GRAPH_AUG, DENSE_ONLY, BM25_ONLY, NO_MERGE, EDGES_OFF, K_LOW, K_DEFAULT]

# -----------------------
# Clean factorial ablations (retrieval x rerank x graph)
# (No prompt/parameter changes; these are just condition labels.)
# -----------------------
CLEAN_ABLATIONS: List[str] = []


def _clean_ablation_id(retrieval: str, rerank: bool, graph: bool) -> str:
    # retrieval in {"ENSEMBLE","BM25_ONLY","VECTOR_ONLY"}
    return f"CLEAN__R={retrieval}__RR={'ON' if rerank else 'OFF'}__G={'ON' if graph else 'OFF'}"


for _retr in ["ENSEMBLE", "BM25_ONLY", "VECTOR_ONLY"]:
    for _rr in [False, True]:
        for _g in [False, True]:
            CLEAN_ABLATIONS.append(_clean_ablation_id(_retr, _rr, _g))


# -----------------------
# Utility functions
# -----------------------
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def truncate_chars(text: str, max_chars: int = 1000) -> str:
    s = str(text or "")
    return s if len(s) <= max_chars else (s[:max_chars] + "... [truncated]")


def paired_contingency(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, int, int]:
    a = a.astype(bool)
    b = b.astype(bool)
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    return n11, n10, n01, n00


def mcnemar_exact_pvalue(n10: int, n01: int, two_sided: bool = True) -> float:
    from math import comb

    n = n10 + n01
    if n == 0:
        return 1.0

    def binom_pmf(k):
        return comb(n, k) * (0.5 ** n)

    k_obs = min(n10, n01)
    p_le = sum(binom_pmf(k) for k in range(0, k_obs + 1))
    if not two_sided:
        return p_le
    return float(min(1.0, 2.0 * p_le))


def paired_permutation_signflip(x, y, n_perm=20000, seed=0, two_sided=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return {"N": 0, "obs_mean_diff": None, "p_value": None}

    d = x - y
    obs = float(d.mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(int(n_perm), d.size), replace=True)
    perm = (signs * d).mean(axis=1)

    if two_sided:
        p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1)
    else:
        p = (np.sum(perm >= obs) + 1) / (len(perm) + 1)

    return {"N": int(d.size), "obs_mean_diff": obs, "p_value": float(p)}


# -----------------------
# Embedding helpers (robust; keep from ablation runner to avoid crashes)
# -----------------------
def _safe_truncate_words(text: str, max_words: int = 100) -> str:
    if not text or not isinstance(text, str):
        return ""
    return " ".join(text.split()[:max_words]).strip()


def _safe_normalize_text(x) -> str:
    if isinstance(x, list):
        x = " ".join(map(str, x))
    if not x or not isinstance(x, str):
        return ""
    return x.strip().lower()


def _safe_embed_sentence_transformer(model, text: str) -> np.ndarray:
    """
    Always returns a 2D numpy array of shape (1, dim).
    Returns shape (1, 0) if embedding fails.
    """
    text = _safe_truncate_words(_safe_normalize_text(text), max_words=100)
    if not text or len(text) < 2:
        return np.zeros((1, 0), dtype=np.float32)

    try:
        with torch.no_grad():
            emb = model.encode(
                [text],                    # enforce batch
                convert_to_tensor=False,   # numpy
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=1
            )
        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.ndim != 2 or emb.shape[0] != 1:
            return np.zeros((1, 0), dtype=np.float32)
        return emb.astype(np.float32, copy=False)
    except Exception:
        return np.zeros((1, 0), dtype=np.float32)


def _safe_cosine_sim(e1: np.ndarray, e2: np.ndarray) -> float:
    if e1 is None or e2 is None:
        return 0.0
    if e1.ndim != 2 or e2.ndim != 2:
        return 0.0
    if e1.shape[1] == 0 or e2.shape[1] == 0:
        return 0.0
    if e1.shape[1] != e2.shape[1]:
        return 0.0
    return float(cosine_similarity(e1, e2)[0, 0])


# -----------------------
# Answer Extraction Functions
# -----------------------
def parse_predicted_option_cop_basic(text: str) -> Tuple[Optional[int], float]:
    """
    Basic regex parsing with confidence estimation.
    Returns (option, confidence)
    """
    if not text:
        return None, 0.0
    t = str(text).strip()

    # High confidence patterns
    m = re.search(r"(?i)(?:correct|final)\s+answer\s*[:\-]?\s*\**\s*(?:option\s+)?([1-4])\b", t)
    if m:
        return int(m.group(1)), 0.95

    m = re.search(r"(?i)\banswer\s*[:\-]?\s*\**\s*(?:option\s+)?([1-4])\b", t)
    if m:
        return int(m.group(1)), 0.85

    m = re.search(r"(?i)\banswer\s*[:\-]?\s*\**\s*([a-d])\b", t)
    if m:
        return OPTION_LETTER_TO_COP[m.group(1).lower()], 0.85

    # Medium confidence patterns
    m = re.search(r"(?i)option\s+([1-4])\s+is\s+(?:correct|right|the\s+answer)", t)
    if m:
        return int(m.group(1)), 0.80

    m = re.search(r"(?i)(?:choose|select)\s+(?:option\s+)?([1-4])\b", t)
    if m:
        return int(m.group(1)), 0.75

    # Low confidence: find in final sentence
    sentences = re.split(r'[.!?]+', t)
    if sentences:
        last_sent = sentences[-1]
        nums = re.findall(r'\b([1-4])\b', last_sent)
        if len(nums) == 1:
            return int(nums[0]), 0.60

    # Lowest confidence: last number anywhere
    nums = re.findall(r"\b([1-4])\b", t)
    if nums:
        return int(nums[-1]), 0.40

    return None, 0.0


def extract_conclusion(text: str, max_words: int = 100) -> str:
    """
    Extract the conclusion/final answer portion of the response.
    """
    if not text:
        return ""

    # Strategy 1: Look for conclusion markers
    conclusion_markers = [
        r'(?i)(?:therefore|thus|hence|in conclusion|to conclude)',
        r'(?i)(?:final answer|correct answer|the answer is)',
        r'(?i)(?:based on (?:this|the (?:above|context)))'
    ]

    for marker in conclusion_markers:
        match = re.search(f'{marker}[^.!?]*[.!?]', text)
        if match:
            conclusion = text[match.start():]
            return ' '.join(conclusion.split()[:max_words])

    # Strategy 2: Take last few sentences
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) >= 3:
        return ' '.join(sentences[-3:])
    return text


# -----------------------
# Main system (matches INITIAL approach + hybrid extraction)
# -----------------------
class RAGGraphMergerWithLLMInitialMatched:
    def __init__(
        self,
        glirel_model_path: str,
        spacy_model_path: str,
        similarity_threshold: float = 0.8,
        chroma_persist_directory: str = "./vectorstore_antiviral_chunk_size_600",
        bm25_persist_directory: str = "bm25_antivir",
        # retrieval knobs (match INITIAL defaults)
        k_vec_default: int = 10,
        k_bm25_default: int = 10,
        k_vec_low: int = 5,
        k_bm25_low: int = 5,
        # rerank knobs (match INITIAL defaults)
        top_n_rerank: int = 10,
        use_reranker: bool = True,
        # LLM (match INITIAL defaults)
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:latest",
        # Extraction LLM (new)
        extraction_model: str = "myaniu/qwen2.5-1m:7b",
        use_llm_extraction: bool = True,
        # storage knobs
        ctx_store_k: int = 4,
        # query behavior (keep existing)
        include_options_in_query: bool = True,
        # relation filtering (match INITIAL default >0.5)
        rel_score_threshold: float = 0.5,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.ctx_store_k = int(ctx_store_k)
        self.include_options_in_query = bool(include_options_in_query)
        self.rel_score_threshold = float(rel_score_threshold)
        self.use_llm_extraction = bool(use_llm_extraction)

        # Match INITIAL: GLiREL on CUDA
        self.model = GLiREL.from_pretrained(glirel_model_path, map_location="cuda")
        self.nlp = spacy.load(spacy_model_path)

        # Match INITIAL: SentenceTransformer with trust_remote_code=True and eval()
        self.embedding_model = SentenceTransformer(
            "NeuML/pubmedbert-base-embeddings",
            trust_remote_code=True,
        )
        self.embedding_model.eval()

        # Match INITIAL: LC embeddings on CPU
        self._lc_embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"batch_size": 1, "normalize_embeddings": True},
        )

        # Vector store and BM25
        self.vectordb = Chroma(
            persist_directory=chroma_persist_directory,
            embedding_function=self._lc_embeddings,
        )
        self._bm25 = BM25SRetriever.from_persisted_directory(
            bm25_persist_directory,
            k=int(k_bm25_default),
        )

        self.k_vec_default = int(k_vec_default)
        self.k_bm25_default = int(k_bm25_default)
        self.k_vec_low = int(k_vec_low)
        self.k_bm25_low = int(k_bm25_low)

        self.top_n_rerank = int(top_n_rerank)
        self.use_reranker = bool(use_reranker)

        self._initialize_llm(ollama_base_url, ollama_model, extraction_model)

        # Labels / constraints: match INITIAL
        self.labels = {
            "glirel_labels": {
                "INDIRECT-DOWNREGULATOR": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "INDIRECT-UPREGULATOR": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "DIRECT-REGULATOR": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "ACTIVATOR": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "INHIBITOR": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "AGONIST": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "AGONIST-ACTIVATOR": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "AGONIST-INHIBITOR": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "ANTAGONIST": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "PRODUCT-OF": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "SUBSTRATE": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "SUBSTRATE_PRODUCT-OF": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                "PART-OF": {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
            }
        }

        self._verb_map = {
            "INDIRECT-DOWNREGULATOR": "indirectly downregulates",
            "INDIRECT-UPREGULATOR": "indirectly upregulates",
            "DIRECT-REGULATOR": "directly regulates",
            "ACTIVATOR": "activates",
            "INHIBITOR": "inhibits",
            "AGONIST": "acts as an agonist for",
            "AGONIST-ACTIVATOR": "acts as an agonist activator for",
            "AGONIST-INHIBITOR": "acts as an agonist inhibitor for",
            "ANTAGONIST": "acts as an antagonist for",
            "PRODUCT-OF": "is produced by",
            "SUBSTRATE": "serves as a substrate for",
            "SUBSTRATE_PRODUCT-OF": "is both substrate and product of",
            "PART-OF": "is part of",
        }

        # Build default retriever (ensemble + rerank) to match INITIAL
        self._build_retriever("ENSEMBLE", self.k_vec_default, self.k_bm25_default, rerank_on=True)

    # -------------------
    # LLM (prompt matches INITIAL verbatim)
    # -------------------
    def _initialize_llm(self, ollama_base_url: str, ollama_model: str, extraction_model: str):
        # Match INITIAL: no extra options seed
        self.llm = ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.0,
            num_ctx=4096,
        )

        # Extraction LLM (new) - using myaniu/qwen2.5-1m:7b
        self.extraction_llm = ChatOllama(
            base_url=ollama_base_url,
            model=extraction_model,
            temperature=0.0,
            num_ctx=8192,
        )

        self.prompt_template = """You are an expert in medicinal chemistry and pharmacology. 
        Below is context information extracted from scientific literature:

        {context}

        Based on this information, answer the following question with options. Follow these rules:
        1. Be precise and factual, only using information from the provided context, quote if necessary
        2. For chemical-gene interactions, specify the type of relationship (e.g., activates, inhibits)
        3. Always include the PubMed source links when available, formatted as: https://pubmed.ncbi.nlm.nih.gov/[PMID]
        4. Make an educated guess if the context is not sufficient and ALWAYS pick and write a number of the answer. Make sure that the correct answer is marked as **Answer** and incorrect answers are not marked at all.
        5. There is ALWAYS a SINGLE correct answer.

        Question: {question}
        Answer: """

        self.qa_prompt = PromptTemplate.from_template(self.prompt_template)
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)

    # -------------------
    # Hybrid Answer Extraction
    # -------------------
    def extract_answer_with_llm(
        self,
        llm_response: str,
        question: str,
        options: List[str]
    ) -> Optional[int]:
        """
        Use an LLM to extract the chosen option from a verbose response.
        """
        extraction_prompt = f"""You are a precise answer extractor. Given a question with multiple choice options and a response, identify which option (1, 2, 3, or 4) the response selected as correct.

Question: {question}

Options:
1) {options[0]}
2) {options[1]}
3) {options[2]}
4) {options[3]}

Response: {llm_response}

Output ONLY the number (1, 2, 3, or 4) of the selected answer. If unclear, output the most likely choice based on the reasoning. Output format: just the single digit.

Answer: """

        try:
            extraction_response = self.extraction_llm.invoke(extraction_prompt).content.strip()
            match = re.search(r'\b([1-4])\b', extraction_response)
            if match:
                return int(match.group(1))
            return None
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return None

    def extract_answer_by_hybrid_similarity(
        self,
        llm_response: str,
        options: List[str],
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> Tuple[Optional[int], Dict[int, Dict[str, float]]]:
        """
        Hybrid approach: semantic similarity + keyword overlap.
        """
        conclusion = extract_conclusion(llm_response)
        if not conclusion:
            return None, {}

        scores = {}
        for i, option in enumerate(options, 1):
            # Semantic similarity
            try:
                with torch.no_grad():
                    conclusion_emb = self.embedding_model.encode(
                        conclusion, convert_to_tensor=True, normalize_embeddings=True
                    )
                    option_emb = self.embedding_model.encode(
                        option, convert_to_tensor=True, normalize_embeddings=True
                    )
                semantic_score = float(
                    torch.nn.functional.cosine_similarity(
                        conclusion_emb.unsqueeze(0),
                        option_emb.unsqueeze(0)
                    )[0]
                )
            except Exception:
                semantic_score = 0.0

            # Keyword overlap
            option_keywords = set(option.lower().split())
            response_keywords = set(conclusion.lower().split())
            keyword_score = (len(option_keywords & response_keywords) / len(option_keywords)) if option_keywords else 0.0

            combined_score = (semantic_weight * semantic_score + keyword_weight * keyword_score)
            scores[i] = {'semantic': semantic_score, 'keyword': keyword_score, 'combined': combined_score}

        if not scores:
            return None, {}

        best_option = max(scores.keys(), key=lambda k: scores[k]['combined'])
        best_score = scores[best_option]['combined']

        if best_score >= 0.25:
            return best_option, scores
        return None, scores
        
    def _is_clean_ablation(self, ablation: str) -> bool:
        return isinstance(ablation, str) and ablation.startswith("CLEAN__R=")

    def _parse_clean_ablation(self, ablation: str) -> Tuple[str, bool, bool]:
        """
        Parses: CLEAN__R=ENSEMBLE__RR=ON__G=OFF
        Returns: (retrieval_mode, rerank_on, graph_on)
        retrieval_mode in {"ENSEMBLE","BM25_ONLY","VECTOR_ONLY"}
        """
        if not self._is_clean_ablation(ablation):
            raise ValueError(f"Not a CLEAN ablation id: {ablation}")

        m = re.match(r"^CLEAN__R=(ENSEMBLE|BM25_ONLY|VECTOR_ONLY)__RR=(ON|OFF)__G=(ON|OFF)$", ablation.strip())
        if not m:
            raise ValueError(f"Malformed CLEAN ablation id: {ablation}")

        retrieval_mode = m.group(1)
        rerank_on = (m.group(2) == "ON")
        graph_on = (m.group(3) == "ON")
        return retrieval_mode, rerank_on, graph_on
        
    def extract_answer_robust(
        self,
        llm_response: str,
        question: str,
        options: List[str],
    ) -> Tuple[Optional[int], str, Dict]:
        """
        Multi-strategy answer extraction with confidence tracking.
        Returns: (predicted_option, method_used, metadata)
        """
        metadata: Dict = {}

        regex_result, regex_confidence = parse_predicted_option_cop_basic(llm_response)
        metadata['regex'] = {'result': regex_result, 'confidence': regex_confidence}
        if regex_confidence >= 0.85:
            return regex_result, 'regex_high_conf', metadata

        sim_result, sim_scores = self.extract_answer_by_hybrid_similarity(llm_response, options)
        metadata['similarity'] = {'result': sim_result, 'scores': sim_scores}

        if sim_result == regex_result and regex_result is not None:
            return regex_result, 'regex_similarity_agree', metadata

        if sim_result is not None and sim_scores.get(sim_result, {}).get('combined', 0) >= 0.4:
            return sim_result, 'similarity_high_conf', metadata

        if self.use_llm_extraction:
            llm_result = self.extract_answer_with_llm(llm_response, question, options)
            metadata['llm_extraction'] = {'result': llm_result}
            if llm_result is not None:
                return llm_result, 'llm_extraction', metadata

        if regex_result is not None:
            return regex_result, 'regex_fallback', metadata
        if sim_result is not None:
            return sim_result, 'similarity_fallback', metadata

        return None, 'all_failed', metadata

    # -------------------
    # Retriever (match INITIAL)
    # -------------------
    def _build_retriever(self, retriever_mode: str, k_vec: int, k_bm25: int, rerank_on: bool = True):
        vectorstore_retriever = self.vectordb.as_retriever(search_kwargs={"k": int(k_vec)})

        keyword_retriever = self._bm25
        keyword_retriever.k = int(k_bm25)

        if retriever_mode == "VECTOR_ONLY":
            base = vectorstore_retriever
        elif retriever_mode == "BM25_ONLY":
            base = keyword_retriever
        elif retriever_mode == "ENSEMBLE":
            base = EnsembleRetriever(
                retrievers=[vectorstore_retriever, keyword_retriever],
                weights=[0.8, 0.2],  # match INITIAL
            )
        else:
            raise ValueError(f"Unknown retriever_mode: {retriever_mode}")

        enable_rerank = bool(self.use_reranker) and bool(rerank_on)
        if enable_rerank:
            compressor = FlashrankRerank(top_n=int(self.top_n_rerank))
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base,
            )
        else:
            self.retriever = base

    def retrieve_relevant_documents(self, query: str) -> List[Dict]:
        if not query or not isinstance(query, str) or not query.strip():
            return []
        try:
            docs = self.retriever.get_relevant_documents(query)
            out = []
            for doc in docs:
                content = getattr(doc, "page_content", None)
                if content and str(content).strip():
                    out.append({"text": content, "metadata": getattr(doc, "metadata", {}) or {}})
            return out
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    # -------------------
    # Entity normalization / similarity (kept robust)
    # -------------------
    def normalize_entity(self, entity_text: str) -> str:
        if not entity_text:
            return ""
        if isinstance(entity_text, list):
            entity_text = " ".join(str(x) for x in entity_text if x)
        normalized = str(entity_text).strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _check_biomedical_patterns(self, entity1: str, entity2: str) -> float:
        patterns = [
            r"\b(alpha|beta|gamma|delta)\b",
            r"\b(receptor|enzyme|protein|gene)\b",
            r"\b\d+[a-z]?\b",
            r"[_-]",
        ]
        e1 = entity1
        e2 = entity2
        for pat in patterns:
            e1 = re.sub(pat, "", e1, flags=re.IGNORECASE)
            e2 = re.sub(pat, "", e2, flags=re.IGNORECASE)
        e1 = re.sub(r"\s+", " ", e1).strip()
        e2 = re.sub(r"\s+", " ", e2).strip()
        if e1 and e2:
            return SequenceMatcher(None, e1, e2).ratio()
        return 0.0

    def calculate_entity_similarity(self, entity1: str, entity2: str) -> float:
        e1_norm = _safe_truncate_words(self.normalize_entity(entity1), max_words=100)
        e2_norm = _safe_truncate_words(self.normalize_entity(entity2), max_words=100)

        if not e1_norm or not e2_norm:
            return 0.0
        if e1_norm == e2_norm:
            return 1.0

        string_sim = SequenceMatcher(None, e1_norm, e2_norm).ratio()
        substring_sim = 0.9 if (e1_norm in e2_norm or e2_norm in e1_norm) else 0.0

        e1_emb = _safe_embed_sentence_transformer(self.embedding_model, e1_norm)
        e2_emb = _safe_embed_sentence_transformer(self.embedding_model, e2_norm)
        cosine_sim = _safe_cosine_sim(e1_emb, e2_emb)

        try:
            pattern_sim = float(self._check_biomedical_patterns(e1_norm, e2_norm))
        except Exception:
            pattern_sim = 0.0

        final_similarity = (
            0.3 * string_sim +
            0.2 * substring_sim +
            0.4 * cosine_sim +
            0.1 * pattern_sim
        )
        return float(max(0.0, min(1.0, final_similarity)))

    # -------------------
    # Relations + graphs (match INITIAL)
    # -------------------
    def extract_relations_from_document(self, text: str, doc_id: str = None, metadata: dict = None) -> List[Dict]:
        if not text or not isinstance(text, str) or not text.strip():
            return []
        if len(text) > 10000:
            text = text[:10000]

        try:
            doc = self.nlp(text)
            tokens = [t.text for t in doc]
            if not tokens or not doc.ents:
                return []

            ner = [[ent.start, (ent.end - 1), ent.label_, ent.text] for ent in doc.ents]
            if not ner:
                return []

            labels_and_constraints = self.labels["glirel_labels"]
            labels_list = list(labels_and_constraints.keys())

            relations = self.model.predict_relations(tokens, labels_list, threshold=0.0, ner=ner, top_k=3)
            relations = constrain_relations_by_entity_type(doc.ents, labels_and_constraints, relations)

            filtered = []
            for rel in relations:
                if float(rel.get("score", 0.0)) <= self.rel_score_threshold:
                    continue

                head_text = rel.get("head_text", "")
                tail_text = rel.get("tail_text", "")
                head_text = " ".join(head_text) if isinstance(head_text, list) else str(head_text)
                tail_text = " ".join(tail_text) if isinstance(tail_text, list) else str(tail_text)

                if not head_text.strip() or not tail_text.strip():
                    continue

                head_ent = next((ent for ent in doc.ents if ent.text == head_text), None)
                tail_ent = next((ent for ent in doc.ents if ent.text == tail_text), None)

                if head_ent and tail_ent and head_ent.label_ == "CHEMICAL" and tail_ent.label_ == "GENE":
                    rel["doc_id"] = doc_id
                    rel["metadata"] = metadata or {}
                    filtered.append(rel)

            return filtered
        except Exception:
            return []

    def create_graph_from_relations(self, relations: List[Dict], doc_id: str = None) -> nx.DiGraph:
        G = nx.DiGraph()
        for rel in relations:
            head_text = " ".join(rel["head_text"]) if isinstance(rel.get("head_text"), list) else rel.get("head_text", "")
            tail_text = " ".join(rel["tail_text"]) if isinstance(rel.get("tail_text"), list) else rel.get("tail_text", "")

            head_norm = self.normalize_entity(head_text)
            tail_norm = self.normalize_entity(tail_text)
            if not head_norm or not tail_norm:
                continue

            metadata = rel.get("metadata", {}) or {}

            G.add_node(
                head_norm,
                original_text=head_text,
                entity_type="CHEMICAL",
                doc_sources={doc_id} if doc_id else set(),
                metadata=metadata,
            )
            G.add_node(
                tail_norm,
                original_text=tail_text,
                entity_type="GENE",
                doc_sources={doc_id} if doc_id else set(),
                metadata=metadata,
            )

            payload = {"label": rel["label"], "score": float(rel["score"]), "doc_id": doc_id, "metadata": metadata}
            if G.has_edge(head_norm, tail_norm):
                G[head_norm][tail_norm].setdefault("relations", []).append(payload)
            else:
                G.add_edge(head_norm, tail_norm, relations=[payload])

        return G

    def find_entity_clusters(self, all_entities: Set[str]) -> List[Set[str]]:
        sim_g = nx.Graph()
        entities = list(all_entities)
        for i, e1 in enumerate(entities):
            sim_g.add_node(e1)
            for e2 in entities[i + 1:]:
                sim = self.calculate_entity_similarity(e1, e2)
                if sim >= self.similarity_threshold:
                    sim_g.add_edge(e1, e2, weight=float(sim))
        return list(nx.connected_components(sim_g))

    def merge_graphs(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        all_entities = set()
        entity_to_graph = {}

        for i, g in enumerate(graphs):
            for node in g.nodes():
                all_entities.add(node)
                entity_to_graph.setdefault(node, []).append(i)

        clusters = self.find_entity_clusters(all_entities) if all_entities else []

        entity_mapping = {}
        canonical_entities = {}

        for cluster in clusters:
            cluster_list = list(cluster)
            canonical = max(cluster_list, key=lambda x: (len(entity_to_graph.get(x, [])), len(x)))

            canonical_entities[canonical] = {
                "original_texts": set(),
                "entity_type": None,
                "doc_sources": set(),
                "metadata": set(),
            }

            for entity in cluster:
                entity_mapping[entity] = canonical
                for gi in entity_to_graph.get(entity, []):
                    g = graphs[gi]
                    if entity in g.nodes():
                        nd = g.nodes[entity]
                        canonical_entities[canonical]["original_texts"].add(nd.get("original_text", entity))
                        canonical_entities[canonical]["entity_type"] = nd.get("entity_type")
                        canonical_entities[canonical]["doc_sources"].update(nd.get("doc_sources", set()))
                        if "metadata" in nd:
                            md = nd["metadata"]
                            canonical_entities[canonical]["metadata"].add(tuple(md.items()) if isinstance(md, dict) else md)

        merged = nx.DiGraph()

        for canonical, info in canonical_entities.items():
            metadata_dicts = []
            for meta in info["metadata"]:
                if isinstance(meta, tuple):
                    metadata_dicts.append(dict(meta))
                elif meta:
                    metadata_dicts.append(meta)

            combined_metadata = {}
            for md in metadata_dicts:
                if md:
                    combined_metadata.update(md)

            merged.add_node(
                canonical,
                original_texts=info["original_texts"],
                entity_type=info["entity_type"],
                doc_sources=info["doc_sources"],
                metadata=combined_metadata,
            )

        for g in graphs:
            for h, t, ed in g.edges(data=True):
                ch = entity_mapping.get(h, h)
                ct = entity_mapping.get(t, t)
                if ch in merged and ct in merged:
                    if merged.has_edge(ch, ct):
                        merged[ch][ct].setdefault("relations", []).extend(ed.get("relations", []))
                    else:
                        merged.add_edge(ch, ct, relations=ed.get("relations", []))

        return merged

    def geometric_mean(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        scores = [float(s) for s in scores]
        if any(s <= 0 for s in scores):
            return 0.0
        return float(np.exp(np.mean(np.log(scores))))

    def _get_relation_verb(self, relation: str) -> str:
        return self._verb_map.get(relation, relation.lower().replace("-", " "))

    def generate_sentences_from_merged_graph(self, merged_graph: nx.DiGraph) -> List[Dict]:
        sentences_with_sources = []

        if merged_graph is None or merged_graph.number_of_edges() == 0:
            return sentences_with_sources

        for head, tail, edge_data in merged_graph.edges(data=True):
            relations = edge_data.get("relations", []) or []
            if not relations:
                continue

            head_data = merged_graph.nodes[head]
            tail_data = merged_graph.nodes[tail]

            head_texts = head_data.get("original_texts", {head})
            tail_texts = tail_data.get("original_texts", {tail})
            head_text = max(head_texts, key=len) if head_texts else head
            tail_text = max(tail_texts, key=len) if tail_texts else tail

            relation_groups = defaultdict(list)
            for rel in relations:
                relation_groups[str(rel.get("label", ""))].append(float(rel.get("score", 0.0)))

            relation_scores = {rtype: self.geometric_mean(scores) for rtype, scores in relation_groups.items() if rtype}

            sources = set()
            for rel in relations:
                md = rel.get("metadata") or {}
                if isinstance(md, dict) and md.get("source"):
                    sources.add(str(md["source"]))

            formatted_sources = [f"https://pubmed.ncbi.nlm.nih.gov/{s}" for s in sources if s]

            doc_count = len(set(rel.get("doc_id") for rel in relations if rel.get("doc_id")))

            if len(relation_scores) == 1:
                rel_type, score = next(iter(relation_scores.items()))
                verb = self._get_relation_verb(rel_type)
                sentence = f"{head_text} {verb} {tail_text} (confidence: {score:.4f}, documents: {doc_count})."
            else:
                verbs = [self._get_relation_verb(rt) for rt in relation_scores.keys()]
                overall = self.geometric_mean(list(relation_scores.values()))
                if len(verbs) == 2:
                    sentence = f"{head_text} {verbs[0]} and {verbs[1]} {tail_text} (confidence: {overall:.4f}, documents: {doc_count})."
                else:
                    sentence = f"{head_text} {', '.join(verbs[:-1])}, and {verbs[-1]} {tail_text} (confidence: {overall:.4f}, documents: {doc_count})."

            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
            sentences_with_sources.append({"sentence": sentence, "sources": formatted_sources})

        return sentences_with_sources

    # -------------------
    # Context building (MATCH INITIAL)
    # -------------------
    def _build_context_initial_style(
        self,
        sentences_with_sources: List[Dict],
        documents: List[Dict],
        include_graph: bool,
        include_docs: bool,
        edges_on: bool,
        merged_graph: Optional[nx.DiGraph] = None,
    ) -> str:
        parts = []

        if include_graph:
            parts.append("=== Generated Knowledge ===")
            if edges_on:
                for item in sentences_with_sources:
                    parts.append(f"- {safe_str(item.get('sentence'))}")
                    srcs = item.get("sources") or []
                    if srcs:
                        parts.append(f"  Sources: {', '.join(srcs)}")
            else:
                chems, genes = [], []
                if merged_graph is not None:
                    for _, data in merged_graph.nodes(data=True):
                        texts = data.get("original_texts", set())
                        disp = max(texts, key=len) if texts else ""
                        if data.get("entity_type") == "CHEMICAL" and disp:
                            chems.append(disp)
                        if data.get("entity_type") == "GENE" and disp:
                            genes.append(disp)
                if chems:
                    parts.append("Chemicals mentioned: " + ", ".join(sorted(set(chems))[:80]) + ".")
                if genes:
                    parts.append("Genes mentioned: " + ", ".join(sorted(set(genes))[:80]) + ".")
                if not chems and not genes:
                    parts.append("(No entities extracted.)")

        if include_docs:
            parts.append("\n=== Original Document Excerpts ===")
            # MATCH INITIAL: include ALL docs (not only first 4) but truncate each to 1000 chars
            for d in (documents or []):
                src = (d.get("metadata") or {}).get("source", "unknown")
                parts.append(f"\nDocument from PMID {src}:")
                parts.append(truncate_chars(d.get("text", ""), 1000))

        return "\n".join(parts)

    # -------------------
    # QA
    # -------------------
    def answer_question(self, question_with_options: str, context: str) -> str:
        return self.llm_chain.run({"question": question_with_options, "context": context}).strip()

    def _make_query(self, question: str, opa: str, opb: str, opc: str, opd: str) -> str:
        if not self.include_options_in_query:
            return question
        return f"{question}\nOptions: {opa} | {opb} | {opc} | {opd}"

    def _question_with_options_text(self, question: str, opa: str, opb: str, opc: str, opd: str) -> str:
        return (
            f"{question}\n\n"
            f"Options:\n"
            f"1) {opa}\n"
            f"2) {opb}\n"
            f"3) {opc}\n"
            f"4) {opd}\n"
        )

    # -------------------
    # Single ablation run
    # -------------------
    def run_ablation(self, ablation: str, question: str, opa: str, opb: str, opc: str, opd: str) -> Dict:
        # Decide retriever, rerank, graph flags (ALWAYS populate these for output)
        retrieval_mode: Optional[str] = None
        rerank_on: Optional[bool] = None
        graph_on: Optional[bool] = None

        if self._is_clean_ablation(ablation):
            retrieval_mode, rerank_on, graph_on = self._parse_clean_ablation(ablation)
            self._build_retriever(
                retrieval_mode,
                self.k_vec_default,
                self.k_bm25_default,
                rerank_on=bool(rerank_on)
            )
        else:
            # Preserve original behavior for legacy ablation labels,
            # but ALSO record the implied retrieval_mode/rerank_on/graph_on.
            if ablation == DENSE_ONLY:
                retrieval_mode = "VECTOR_ONLY"
                rerank_on = False
                self._build_retriever(retrieval_mode, self.k_vec_default, self.k_bm25_default, rerank_on=rerank_on)

            elif ablation == BM25_ONLY:
                retrieval_mode = "BM25_ONLY"
                rerank_on = False
                self._build_retriever(retrieval_mode, self.k_vec_default, self.k_bm25_default, rerank_on=rerank_on)

            elif ablation == K_LOW:
                retrieval_mode = "ENSEMBLE"
                rerank_on = True
                self._build_retriever(retrieval_mode, self.k_vec_low, self.k_bm25_low, rerank_on=rerank_on)

            else:
                # FULL_SYSTEM, NO_GRAPH_AUG, NO_MERGE, EDGES_OFF, TOPK_K_DEFAULT, etc.
                retrieval_mode = "ENSEMBLE"
                rerank_on = True
                self._build_retriever(retrieval_mode, self.k_vec_default, self.k_bm25_default, rerank_on=rerank_on)

            # Graph flag for legacy ablations
            graph_on = (ablation != NO_GRAPH_AUG)

        # Downstream flags
        include_graph = bool(graph_on)
        include_docs = True
        merging_on = (ablation != NO_MERGE)
        edges_on = (ablation != EDGES_OFF)

        query = self._make_query(question, opa, opb, opc, opd)
        documents = self.retrieve_relevant_documents(query)

        # Store top-4 docs for CSV columns
        all_ctx_texts = [safe_str(d.get("text", "")) for d in documents]
        ctx_topk = all_ctx_texts[: self.ctx_store_k]
        while len(ctx_topk) < self.ctx_store_k:
            ctx_topk.append("")

        # No docs: still call LLM with empty context (match INITIAL behavior)
        if not documents:
            q_text = self._question_with_options_text(question, opa, opb, opc, opd)
            context = self._build_context_initial_style(
                sentences_with_sources=[],
                documents=[],
                include_graph=include_graph,
                include_docs=include_docs,
                edges_on=edges_on,
                merged_graph=None,
            )
            model_answer = self.answer_question(q_text, context)

            pred, method, extract_meta = self.extract_answer_robust(
                model_answer, question, [opa, opb, opc, opd]
            )

            return {
                "ablation": ablation,
                "query": query,
                "n_docs": 0,
                "ctx_1": ctx_topk[0],
                "ctx_2": ctx_topk[1],
                "ctx_3": ctx_topk[2],
                "ctx_4": ctx_topk[3],
                "context": context,
                "model_answer": model_answer,
                "predicted_cop": pred,
                "extraction_method": method,
                "extraction_metadata": json.dumps(extract_meta),
                "graph_num_nodes": 0,
                "graph_num_edges": 0,
                # extra clarity columns (NOW always filled)
                "retrieval_mode": retrieval_mode,
                "rerank_on": bool(rerank_on),
                "graph_on": bool(graph_on),
            }

        # Graph path
        graphs: List[nx.DiGraph] = []
        merged_graph: Optional[nx.DiGraph] = None
        sentences_with_sources: List[Dict] = []

        if include_graph:
            # Build per-doc graphs ONLY when graph is used
            for idx, doc in enumerate(documents):
                doc_id = f"doc_{idx}"
                rels = self.extract_relations_from_document(
                    text=doc.get("text", ""),
                    doc_id=doc_id,
                    metadata=doc.get("metadata", {}),
                )
                g = self.create_graph_from_relations(rels, doc_id=doc_id)
                graphs.append(g)

            # Merge graphs (or not) ONCE (bugfix: remove duplicate merge)
            if merging_on:
                merged_graph = self.merge_graphs(graphs)
            else:
                merged_graph = nx.DiGraph()
                for g in graphs:
                    merged_graph = nx.compose(merged_graph, g)

            # Generate sentences
            if merged_graph is not None and merged_graph.number_of_edges() > 0 and edges_on:
                sentences_with_sources = self.generate_sentences_from_merged_graph(merged_graph)

        # Build context (MATCH INITIAL: include ALL docs in prompt)
        context = self._build_context_initial_style(
            sentences_with_sources=sentences_with_sources,
            documents=documents,
            include_graph=include_graph,
            include_docs=include_docs,
            edges_on=edges_on,
            merged_graph=merged_graph,
        )

        q_text = self._question_with_options_text(question, opa, opb, opc, opd)
        model_answer = self.answer_question(q_text, context)

        pred, method, extract_meta = self.extract_answer_robust(
            model_answer, question, [opa, opb, opc, opd]
        )

        return {
            "ablation": ablation,
            "query": query,
            "n_docs": int(len(documents)),
            "ctx_1": ctx_topk[0],
            "ctx_2": ctx_topk[1],
            "ctx_3": ctx_topk[2],
            "ctx_4": ctx_topk[3],
            "context": context,
            "model_answer": model_answer,
            "predicted_cop": pred,
            "extraction_method": method,
            "extraction_metadata": json.dumps(extract_meta),
            "graph_num_nodes": int(merged_graph.number_of_nodes()) if merged_graph is not None else 0,
            "graph_num_edges": int(merged_graph.number_of_edges()) if merged_graph is not None else 0,
            # extra clarity columns (NOW always filled)
            "retrieval_mode": retrieval_mode,
            "rerank_on": bool(rerank_on),
            "graph_on": bool(graph_on),
        }


# -----------------------
# Batch runner
# -----------------------
def run_file(
    input_csv: str,
    output_csv: str,
    system: RAGGraphMergerWithLLMInitialMatched,
    ablations: List[str],
    seed: int = 0,
    checkpoint_every: int = 1,
):
    df = pd.read_csv(input_csv)

    # Handle both naming conventions for options
    option_columns = {
        'question': 'question',
        'option_a': None,
        'option_b': None,
        'option_c': None,
        'option_d': None,
        'correct_option': None
    }

    if 'option_a' in df.columns:
        option_columns['option_a'] = 'option_a'
        option_columns['option_b'] = 'option_b'
        option_columns['option_c'] = 'option_c'
        option_columns['option_d'] = 'option_d'
    elif 'opa' in df.columns:
        option_columns['option_a'] = 'opa'
        option_columns['option_b'] = 'opb'
        option_columns['option_c'] = 'opc'
        option_columns['option_d'] = 'opd'
    else:
        raise ValueError(f"Missing option columns in {input_csv}. Expected either 'option_a/b/c/d' or 'opa/b/c/d'")

    if 'correct_option' in df.columns:
        option_columns['correct_option'] = 'correct_option'
    elif 'cop' in df.columns:
        option_columns['correct_option'] = 'cop'
    else:
        print(f"Warning: No correct_option or cop column found in {input_csv}. Evaluation will not be possible.")
        option_columns['correct_option'] = None

    if 'question' not in df.columns:
        raise ValueError(f"Missing required column 'question' in {input_csv}")

    # Resume support
    existing_rows = []
    processed_pairs = set()  # (row_id, ablation)

    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing_rows = existing_df.to_dict('records')

            # Defensive: if file is empty or missing keys, don't crash
            if 'row_id' in existing_df.columns and 'ablation' in existing_df.columns:
                processed_pairs = set(zip(existing_df['row_id'], existing_df['ablation']))

            print(f"Loaded {len(existing_rows)} existing results from {output_csv}")
            print(f"Already processed: {len(processed_pairs)} (row_id, ablation) pairs")
        except Exception as e:
            print(f"Could not load existing results: {e}")
            existing_rows = []
            processed_pairs = set()

    rows = existing_rows.copy()

    pbar = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing questions",
        position=0,
        leave=True,
        ncols=100
    )

    questions_processed = 0

    for i, row in pbar:
        q = safe_str(row.get('question'))
        oa = safe_str(row.get(option_columns['option_a']))
        ob = safe_str(row.get(option_columns['option_b']))
        oc = safe_str(row.get(option_columns['option_c']))
        od = safe_str(row.get(option_columns['option_d']))

        gold = None
        if option_columns['correct_option']:
            gold = row.get(option_columns['correct_option'], None)

        try:
            gold_int = int(gold) if gold is not None and not (isinstance(gold, float) and np.isnan(gold)) else None
        except Exception:
            gold_int = None

        question_had_new_results = False

        for ab_idx, ab in enumerate(ablations):
            if (i, ab) in processed_pairs:
                pbar.set_description(f"Q{i+1}/{len(df)} | Ablation {ab_idx+1}/{len(ablations)}: {ab[:20]} [SKIP]")
                continue

            pbar.set_description(f"Q{i+1}/{len(df)} | Ablation {ab_idx+1}/{len(ablations)}: {ab[:20]}")

            set_global_seed(seed + i)
            out = system.run_ablation(ab, q, oa, ob, oc, od)

            out_row = {
                "row_id": i,
                "ablation": out["ablation"],
                "question": q,
                "option_a": oa,
                "option_b": ob,
                "option_c": oc,
                "option_d": od,
                "correct_option": gold_int,
                "query": out["query"],
                "n_docs": out["n_docs"],
                "graph_num_nodes": out["graph_num_nodes"],
                "graph_num_edges": out["graph_num_edges"],
                "predicted_cop": out["predicted_cop"],
                "extraction_method": out["extraction_method"],
                "extraction_metadata": out["extraction_metadata"],
                "is_correct": (int(out["predicted_cop"] == gold_int)
                               if gold_int is not None and out["predicted_cop"] is not None else None),
                "ctx_1": out["ctx_1"],
                "ctx_2": out["ctx_2"],
                "ctx_3": out["ctx_3"],
                "ctx_4": out["ctx_4"],
                "model_answer": out["model_answer"],
                "context": out["context"],
                # added clarity columns (won't break anything if you ignore them)
                "retrieval_mode": out.get("retrieval_mode"),
                "rerank_on": out.get("rerank_on"),
                "graph_on": out.get("graph_on"),
            }

            rows.append(out_row)
            processed_pairs.add((i, ab))
            question_had_new_results = True

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if question_had_new_results:
            questions_processed += 1
            if questions_processed % checkpoint_every == 0:
                out_df = pd.DataFrame(rows)
                out_df.to_csv(output_csv, index=False)
                pbar.set_postfix({"saved": f"{len(rows)} rows"})

    pbar.close()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"\nFinal results saved to {output_csv}: {len(rows)} total rows")
    return out_df


# -----------------------
# Evaluation
# -----------------------
def eval_stats(output_csv: str, baseline: str = FULL_SYSTEM):
    df = pd.read_csv(output_csv)

    if "correct_option" not in df.columns:
        raise ValueError("Output CSV must include correct_option to evaluate.")
    if "predicted_cop" not in df.columns:
        raise ValueError("Output CSV must include predicted_cop to evaluate.")
    if "ablation" not in df.columns:
        raise ValueError("Output CSV must include ablation.")

    if "extraction_method" in df.columns:
        print("\nExtraction Method Distribution:")
        method_counts = df["extraction_method"].value_counts()
        for method, count in method_counts.items():
            pct = 100 * count / len(df)
            print(f"  {method:25s}: {count:5d} ({pct:5.1f}%)")
        print()

    accs = []
    for ab, g in df.groupby("ablation"):
        g2 = g.dropna(subset=["correct_option", "predicted_cop"])
        if len(g2) == 0:
            acc = np.nan
            n = 0
        else:
            acc = float(np.mean(g2["correct_option"].astype(int) == g2["predicted_cop"].astype(int)))
            n = int(len(g2))
        accs.append((ab, n, acc))

    accs = sorted(accs, key=lambda x: (x[0] != baseline, x[0]))
    print("Accuracy by ablation:")
    for ab, n, acc in accs:
        print(f"  {ab:22s}  N={n:5d}  acc={acc:.4f}" if np.isfinite(acc) else f"  {ab:22s}  N={n:5d}  acc=NA")

    base = df[df["ablation"] == baseline].copy()
    if base.empty:
        print(f"\nNo baseline rows found for ablation='{baseline}'. Skipping paired tests.")
        return

    base = base.set_index(["row_id"])
    print("\nPaired tests vs baseline (McNemar exact on correctness):")
    for ab in sorted(df["ablation"].unique()):
        if ab == baseline:
            continue
        other = df[df["ablation"] == ab].copy().set_index(["row_id"])

        joined = base.join(other, lsuffix="_base", rsuffix="_oth", how="inner")
        joined = joined.dropna(subset=["is_correct_base", "is_correct_oth"])
        if joined.empty:
            print(f"  {ab:22s}  N=0  p=NA")
            continue

        a = joined["is_correct_base"].astype(int).values
        b = joined["is_correct_oth"].astype(int).values
        n11, n10, n01, n00 = paired_contingency(a == 1, b == 1)
        p = mcnemar_exact_pvalue(n10=n10, n01=n01, two_sided=True)
        print(f"  {ab:22s}  N={len(joined):5d}  n10={n10:4d}  n01={n01:4d}  p={p:.6f}")

    print("\nPermutation sign-flip test on per-question correctness diff (other - baseline):")
    base_corr = base["is_correct"].astype(float)
    for ab in sorted(df["ablation"].unique()):
        if ab == baseline:
            continue
        other = df[df["ablation"] == ab].copy().set_index(["row_id"])
        joined = pd.concat([base_corr.rename("base"), other["is_correct"].astype(float).rename("oth")], axis=1).dropna()
        if joined.empty:
            print(f"  {ab:22s}  N=0  p=NA")
            continue
        res = paired_permutation_signflip(joined["oth"].values, joined["base"].values, n_perm=20000, seed=0, two_sided=True)
        print(f"  {ab:22s}  N={res['N']:5d}  mean_diff={res['obs_mean_diff']:+.6f}  p={res['p_value']:.6f}")


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--glirel_model_path", type=str, required=False, default="./Drugprot_REL_model")
    parser.add_argument("--spacy_model_path", type=str, required=False, default="./NER_Model/model-best")
    parser.add_argument("--chroma_dir", type=str, required=False, default="./vectorstore_antiviral_chunk_size_600")
    parser.add_argument("--bm25_dir", type=str, required=False, default="bm25_antivir")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--k_vec_default", type=int, default=10)
    parser.add_argument("--k_bm25_default", type=int, default=10)
    parser.add_argument("--k_vec_low", type=int, default=5)
    parser.add_argument("--k_bm25_low", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=1,
                        help="Save results every N questions (default: 1 = after each question)")
    parser.add_argument("--top_n_rerank", type=int, default=10)

    parser.add_argument("--ctx_store_k", type=int, default=4)
    parser.add_argument("--no_reranker", action="store_true", help="Disable Flashrank reranker (deviates from INITIAL).")

    parser.add_argument("--extraction_model", type=str, default="myaniu/qwen2.5-1m:7b", help="Model for answer extraction")
    parser.add_argument("--no_llm_extraction", action="store_true", help="Disable LLM extraction fallback")

    parser.add_argument("--ablations", type=str, default=",".join(ABLATIONS))
    parser.add_argument("--eval_stats", action="store_true")
    parser.add_argument("--baseline", type=str, default=FULL_SYSTEM)

    args = parser.parse_args()

    if args.eval_stats:
        if not args.output_csv:
            raise ValueError("--output_csv is required for --eval_stats")
        eval_stats(args.output_csv, baseline=args.baseline)
        return

    if not args.input_csv or not args.output_csv:
        raise ValueError("--input_csv and --output_csv are required")

    set_global_seed(args.seed)

    system = RAGGraphMergerWithLLMInitialMatched(
        glirel_model_path=args.glirel_model_path,
        spacy_model_path=args.spacy_model_path,
        similarity_threshold=0.8,
        chroma_persist_directory=args.chroma_dir,
        bm25_persist_directory=args.bm25_dir,
        k_vec_default=args.k_vec_default,
        k_bm25_default=args.k_bm25_default,
        k_vec_low=args.k_vec_low,
        k_bm25_low=args.k_bm25_low,
        top_n_rerank=args.top_n_rerank,
        use_reranker=(not args.no_reranker),
        extraction_model=args.extraction_model,
        use_llm_extraction=(not args.no_llm_extraction),
        ctx_store_k=args.ctx_store_k,
        include_options_in_query=True,
        rel_score_threshold=0.5,
    )

    ablations = [x.strip() for x in args.ablations.split(",") if x.strip()]
    run_file(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        system=system,
        ablations=ablations,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
