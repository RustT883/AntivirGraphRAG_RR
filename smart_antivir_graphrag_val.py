# RAG Graph Merger with LLM Integration for CHEMICAL-GENE Relations
import pandas as pd
import spacy
import scispacy
import os
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
from langchain_community.retrievers import BM25SRetriever
from glirel import GLiREL
from glirel.modules.utils import constrain_relations_by_entity_type
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_community.document_loaders import CSVLoader
from ragas import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import DirectoryLoader
import gc
from langchain.retrievers import EnsembleRetriever
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import Faithfulness
import asyncio
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import LLMContextPrecisionWithReference
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics import ContextEntityRecall
from ragas.metrics import NoiseSensitivity
from ragas.metrics import ResponseRelevancy
from ragas.metrics import LLMContextRecall
import warnings
import logging

torch.cuda.empty_cache()
gc.collect()
device = torch.device("cuda")

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*INFO:.*")
warnings.filterwarnings("ignore")

class SmartRAGGraphMerger:
    def __init__(self, glirel_model_path: str, spacy_model_path: str, similarity_threshold: float = 0.8):
        """
        Initialize the Smart RAG Graph Merger with conditional graph processing.
        
        Args:
            glirel_model_path: Path to trained GLiREL model
            spacy_model_path: Path to trained spaCy NER model
            similarity_threshold: Threshold for merging similar entities (default: 0.8)
        """
        self.model = GLiREL.from_pretrained(glirel_model_path, map_location='cuda')
        self.nlp = spacy.load(spacy_model_path)
        self.similarity_threshold = similarity_threshold
        
        # Load medembed-large model
        self.embedding_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings', trust_remote_code=True)
        self.embedding_model.eval()
        
        # Initialize document retriever
        self.initialize_retriever()
        
        # Initialize LLM
        self.initialize_llm()
        
        # Define relation labels and constraints
        self.labels = {
            'glirel_labels': {
                'INDIRECT-DOWNREGULATOR': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'INDIRECT-UPREGULATOR': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'DIRECT-REGULATOR': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'ACTIVATOR': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'INHIBITOR': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'AGONIST': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'AGONIST-ACTIVATOR': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'AGONIST-INHIBITOR': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'ANTAGONIST': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'PRODUCT-OF': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'SUBSTRATE': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'SUBSTRATE_PRODUCT-OF': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]},
                'PART-OF': {"allowed_head": ["CHEMICAL"], "allowed_tail": ["GENE"]}
            }
        }
        
        # Define graph-triggering keywords
        self.graph_keywords = {
            # Direct interaction terms
            'interaction', 'interactions', 'interact', 'interacts', 'binding', 'binds', 'bind',
            'target', 'targets', 'targeting', 'targeted',
            'mechanism', 'mechanisms', 'pathway', 'pathways',
            'inhibit', 'inhibits', 'inhibition', 'inhibitor', 'inhibitors',
            'activate', 'activates', 'activation', 'activator', 'activators',
            'regulate', 'regulates', 'regulation', 'regulator', 'regulators',
            'modulate', 'modulates', 'modulation', 'modulator', 'modulators',
            
            # Specific relationship terms
            'agonist', 'agonists', 'antagonist', 'antagonists',
            'substrate', 'substrates', 'product', 'products',
            'upregulate', 'upregulates', 'upregulation', 'upregulator',
            'downregulate', 'downregulates', 'downregulation', 'downregulator',
            
            # Drug-related terms
            'drug', 'drugs', 'compound', 'compounds', 'chemical', 'chemicals',
            'pharmaceutical', 'pharmaceuticals', 'medicine', 'medicines',
            'therapeutic', 'therapeutics', 'treatment', 'treatments',
            
            # Biological entity terms
            'protein', 'proteins', 'gene', 'genes', 'enzyme', 'enzymes',
            'receptor', 'receptors', 'channel', 'channels',
            'transporter', 'transporters', 'carrier', 'carriers',
            
            # Relationship indicators
            'relationship', 'relationships', 'connection', 'connections',
            'effect', 'effects', 'affects', 'influence', 'influences',
            'response', 'responses', 'activity', 'activities',
            
            # Mechanism-specific terms
            'signaling', 'cascade', 'network', 'networks',
            'complex', 'complexes', 'assembly', 'assemblies',
            'catalysis', 'catalyzes', 'enzymatic', 'kinetic', 'kinetics'
        }
    
    def should_use_graph_method(self, query: str) -> bool:
        """
        Determine if the query requires graph-based processing.
        
        Args:
            query: The user's query
            
        Returns:
            True if graph method should be used, False for basic RAG
        """
        if not query or not isinstance(query, str):
            return False
        
        query_lower = query.lower()
        
        # Check for graph-triggering keywords
        for keyword in self.graph_keywords:
            if keyword in query_lower:
                return True
        
        # Additional pattern matching for complex queries
        graph_patterns = [
            r'how\s+does\s+\w+\s+(affect|influence|interact|bind|target)',
            r'what\s+is\s+the\s+(mechanism|pathway|interaction|relationship)',
            r'which\s+(drugs|compounds|chemicals)\s+(target|inhibit|activate|bind)',
            r'(drug|compound|chemical)\s+(target|mechanism|interaction|pathway)',
            r'(protein|gene|enzyme|receptor)\s+(interaction|binding|regulation)',
            r'(inhibitor|activator|agonist|antagonist)\s+of',
            r'(upregulat|downregulat|modulat|regulat)\w*\s+(by|of)',
        ]
        
        for pattern in graph_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def initialize_retriever(self):
        """Initialize the document retrieval system."""
        # Initialize embeddings and vector store
        core_embeddings_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'device': "cpu", 'trust_remote_code': True},
            encode_kwargs={'batch_size': 1, 'normalize_embeddings': True}
        )
        
        persist_directory = "./vectorstore_antiviral_chunk_size_600"

        keyword_retriever = BM25SRetriever.from_persisted_directory(
                "bm25_antivir",
                k=10
        )

        self.vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=core_embeddings_model
        )
        
        # Configure retriever with reranker
        vectorstore_retriever = self.vectordb.as_retriever(search_kwargs={'k': 10})
        
        
        self.retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, keyword_retriever],
            weights=[0.7, 0.3]
        )

        #compressor = FlashrankRerank(top_n=10, model="ms-marco-MiniLM-L-12-v2")
        #self.retriever = ContextualCompressionRetriever(
        #    base_compressor=compressor,
        #    base_retriever=ensemble_retriever
        #)
    
    def initialize_llm(self):
        """Initialize the LLM for question answering."""
        self.llm = ChatOllama(
            base_url="http://localhost:11434",
            model="qwen2.5:latest",
            temperature=0.0,
            num_ctx=4096
        )
        
        # Define document prompt for source formatting
        self.document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Content:{page_content}\nSource:https://pubmed.ncbi.nlm.nih.gov/{source}\n",
        )
        
        # Define prompt template for graph-based responses
        self.graph_prompt_template = """You are an expert in medicinal chemistry and pharmacology.  
        Below is context information extracted from scientific literature and a knowledge graph:  

        {context}  

        Based on this information, answer the following question with options. Follow these rules:  
        1. You MUST select ONLY one answer, even if multiple options seem plausible. 
        2. Be precise and factual, only using information from the provided context. Quote key phrases if needed.  
        3. For chemical-gene interactions, specify the type of relationship (e.g., activates, inhibits).  
        4. Include PubMed links when available: https://pubmed.ncbi.nlm.nih.gov/[PMID].  
        5. Interpret the question carefully:  
           - If asking about "most common," prioritize explicit frequency data.  
           - If asking about "most severe," prioritize severity descriptions.  
           - If unclear, use pharmacological reasoning.  
        6. Do not assume frequently retrieved = most correct. Verify from context.   
        7. If context is insufficient, make the most plausible guess.  

        Question: {question}  
        Answer: """
                
                # Define prompt template for basic RAG responses
        self.basic_prompt_template = """You are an expert in medicinal chemistry and pharmacology.  
    Below is context information extracted from scientific literature:  

        {context}  

        Based on this information, answer the following question with options. Follow these rules:  
        1. You MUST select ONLY one answer, even if multiple options seem plausible. 
        2. Be precise and factual, only using information from the provided context. Quote key phrases if needed.  
        3. Include PubMed links when available: https://pubmed.ncbi.nlm.nih.gov/[PMID].  
        4. Interpret the question carefully:  
           - If asking about "most common," prioritize explicit frequency data.  
           - If asking about "most severe," prioritize severity descriptions.  
           - If unclear, use pharmacological reasoning.  
        5. Do not assume frequently retrieved = most correct. Verify from context.  
        6. If context is insufficient, make the most plausible guess.  

        Question: {question}  
        Answer: """
        
        self.graph_qa_prompt = PromptTemplate.from_template(self.graph_prompt_template)
        self.basic_qa_prompt = PromptTemplate.from_template(self.basic_prompt_template)
        
        self.graph_llm_chain = LLMChain(llm=self.llm, prompt=self.graph_qa_prompt)
        self.basic_llm_chain = LLMChain(llm=self.llm, prompt=self.basic_qa_prompt)
    
    def retrieve_relevant_documents(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query with their metadata.
        
        Args:
            query: The search query
            
        Returns:
            List of dictionaries with 'text' and 'metadata'
        """
        # Validate input
        if not query or not isinstance(query, str) or not query.strip():
            return []
        
        try:
            # Get documents with error handling
            docs = self.retriever.get_relevant_documents(query)
            
            # Process documents with additional validation
            valid_docs = []
            for doc in docs:
                if hasattr(doc, 'page_content') and doc.page_content.strip():
                    doc_dict = {
                        'text': doc.page_content,
                        'metadata': getattr(doc, 'metadata', {})
                    }
                    valid_docs.append(doc_dict)
            
            return valid_docs
        
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []

    def process_basic_rag_query(self, query: str) -> Tuple[str, dict]:
        """
        Process query using basic RAG without graph construction.
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (answer, document_texts)
        """
        # Initialize document texts
        document_texts = {
            'page_content_graph_1': '',
            'page_content_graph_2': '',
            'page_content_graph_3': '',
            'page_content_graph_4': '',
        }
        
        # Retrieve relevant documents
        print(f"Processing with Basic RAG for query: {query}")
        documents = self.retrieve_relevant_documents(query)
        
        # Store document texts
        for i, doc in enumerate(documents[:4]):
            col_name = f'page_content_graph_{i+1}'
            document_texts[col_name] = doc['text'][:10000] if 'text' in doc else ''
        
        if not documents:
            return "I couldn't find sufficient information to answer this question.", document_texts
        
        # Create context from documents
        context_parts = ["=== Document Context ==="]
        for doc in documents:
            text = doc.get('text', '')
            if len(text) > 2000:  # Larger context for basic RAG
                text = text[:2000] + "... [truncated]"
            
            source = doc.get('metadata', {}).get('source', 'unknown')
            context_parts.append(f"\nDocument from PMID {source}:")
            context_parts.append(text)
        
        context = "\n".join(context_parts)
        
        # Generate answer using basic RAG chain
        answer = self.basic_llm_chain.run({
            "question": query,
            "context": context
        }).strip()
        
        return answer, document_texts

    def calculate_entity_similarity(self, entity1: str, entity2: str) -> float:
        """
        Calculate similarity between two entities using multiple methods.
        Handles long strings by truncating them before embedding.
        """
        # Normalize and truncate entities (keep first 512 words to prevent OOM)
        def safe_truncate(text, max_words=512):
            words = text.split()[:max_words]
            return ' '.join(words)
        
        e1_norm = self.normalize_entity(entity1)
        e2_norm = self.normalize_entity(entity2)
        
        # Further truncate very long strings
        e1_norm = safe_truncate(e1_norm)
        e2_norm = safe_truncate(e2_norm)

        # Exact match
        if e1_norm == e2_norm:
            return 1.0
        
        # String similarity
        string_sim = SequenceMatcher(None, e1_norm, e2_norm).ratio()
        substring_sim = 0.9 if (e1_norm in e2_norm or e2_norm in e1_norm) else 0.0

        # Initialize cosine similarity
        cosine_sim = 0.0
        
        # Only proceed with embeddings if we have valid text
        if len(e1_norm) > 0 and len(e2_norm) > 0:
            try:
                with torch.no_grad():
                    # Get embeddings
                    e1_embed = self.embedding_model.encode(
                        e1_norm,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                    e2_embed = self.embedding_model.encode(
                        e2_norm,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                    
                    # Ensure we have valid 1D embeddings
                    if e1_embed.dim() != 1 or e2_embed.dim() != 1:
                        if e1_embed.dim() == 2:
                            e1_embed = e1_embed.squeeze(0)
                        if e2_embed.dim() == 2:
                            e2_embed = e2_embed.squeeze(0)
                    
                    # Verify we have 1D tensors
                    if e1_embed.dim() == 1 and e2_embed.dim() == 1:
                        cosine_sim = torch.nn.functional.cosine_similarity(
                            e1_embed.unsqueeze(0),
                            e2_embed.unsqueeze(0),
                            dim=1
                        ).item()
                        
            except Exception as e:
                print(f"Error in embedding calculation: {str(e)}")
                cosine_sim = 0.0

        # Pattern similarity
        try:
            pattern_sim = self._check_biomedical_patterns(e1_norm, e2_norm)
        except Exception as e:
            print(f"Error in pattern matching: {str(e)}")
            pattern_sim = 0.0

        # Weighted combination
        final_similarity = (
            0.3 * string_sim +
            0.2 * substring_sim +
            0.4 * cosine_sim +
            0.1 * pattern_sim
        )
        
        return max(0.0, min(1.0, final_similarity))
    
    def _check_biomedical_patterns(self, entity1: str, entity2: str) -> float:
        """Check for common biomedical naming patterns."""
        patterns = [
            r'\b(alpha|beta|gamma|delta)\b',
            r'\b(receptor|enzyme|protein|gene)\b',
            r'\b\d+[a-z]?\b',
            r'[_-]',
        ]
        
        e1_clean = entity1
        e2_clean = entity2
        
        for pattern in patterns:
            e1_clean = re.sub(pattern, '', e1_clean, flags=re.IGNORECASE)
            e2_clean = re.sub(pattern, '', e2_clean, flags=re.IGNORECASE)
        
        e1_clean = re.sub(r'\s+', ' ', e1_clean).strip()
        e2_clean = re.sub(r'\s+', ' ', e2_clean).strip()
        
        if e1_clean and e2_clean:
            return SequenceMatcher(None, e1_clean, e2_clean).ratio()
        return 0.0

    def normalize_entity(self, entity_text: str) -> str:
        """Normalize entity text for consistent matching."""
        if not entity_text:
            return ""
        
        if isinstance(entity_text, list):
            entity_text = ' '.join(str(item) for item in entity_text if item)
        
        # Convert to string and strip whitespace
        normalized = str(entity_text).strip().lower()
        
        # Remove excessive whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Return empty string if nothing meaningful remains
        if not normalized or normalized.isspace():
            return ""
        
        return normalized

    def extract_relations_from_document(self, text: str, doc_id: str = None, metadata: dict = None) -> List[Dict]:
        """
        Extract relations from a single document.
        
        Args:
            text: Document text
            doc_id: Optional document identifier
            metadata: Document metadata including PMID
            
        Returns:
            List of relation dictionaries
        """
        # Input validation
        if not text or not isinstance(text, str) or not text.strip():
            return []
        
        # Truncate very long texts to prevent memory issues
        if len(text) > 10000:
            text = text[:10000]
        
        try:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            ner = [[ent.start, (ent.end - 1), ent.label_, ent.text] for ent in doc.ents]
            
            # Additional validation for NER results
            if not ner:
                return []
            
            labels_and_constraints = self.labels["glirel_labels"]
            labels_list = list(labels_and_constraints.keys())
            
            relations = self.model.predict_relations(tokens, labels_list, threshold=0.0, ner=ner, top_k=3)
            relations = constrain_relations_by_entity_type(doc.ents, labels_and_constraints, relations)
            
            # Filter relations with score > 0.5 and ensure directionality (CHEMICAL -> GENE)
            filtered_relations = []
            for rel in relations:
                if rel['score'] > 0.5:
                    # Safely extract head and tail text
                    head_text = rel.get('head_text', '')
                    tail_text = rel.get('tail_text', '')
                    
                    if isinstance(head_text, list):
                        head_text = ' '.join(str(item) for item in head_text if item)
                    if isinstance(tail_text, list):
                        tail_text = ' '.join(str(item) for item in tail_text if item)
                    
                    # Validate that we have meaningful text
                    if not head_text or not tail_text:
                        continue
                    
                    head_ent = next((ent for ent in doc.ents if ent.text == head_text), None)
                    tail_ent = next((ent for ent in doc.ents if ent.text == tail_text), None)
                    
                    if (head_ent and tail_ent and
                        head_ent.label_ == "CHEMICAL" and
                        tail_ent.label_ == "GENE"):
                        rel['doc_id'] = doc_id
                        rel['metadata'] = metadata or {}
                        filtered_relations.append(rel)
            
            return filtered_relations
            
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
            return []

    def clear_gpu_memory(self):
        """Clear GPU memory to prevent accumulation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def create_graph_from_relations(self, relations: List[Dict], doc_id: str = None) -> nx.DiGraph:
        G = nx.DiGraph()
        
        for rel in relations:
            head_text = ' '.join(rel['head_text']) if isinstance(rel['head_text'], list) else rel['head_text']
            tail_text = ' '.join(rel['tail_text']) if isinstance(rel['tail_text'], list) else rel['tail_text']
            
            head_norm = self.normalize_entity(head_text)
            tail_norm = self.normalize_entity(tail_text)
            
            # Initialize node attributes with proper defaults
            default_attrs = {
                'original_texts': set(),
                'entity_type': 'UNKNOWN',
                'doc_sources': set(),
                'metadata': {}
            }
            
            # Add nodes with attributes, preserving existing ones if present
            for node, text, ent_type in [(head_norm, head_text, 'CHEMICAL'), 
                                        (tail_norm, tail_text, 'GENE')]:
                if not G.has_node(node):
                    G.add_node(node, **default_attrs)
                
                # Update node attributes
                node_data = G.nodes[node]
                node_data['original_texts'].add(text)
                node_data['entity_type'] = ent_type
                if doc_id:
                    node_data['doc_sources'].add(doc_id)
                if 'metadata' in rel:
                    node_data['metadata'].update(rel.get('metadata', {}))
            
            # Add or update edge
            if G.has_edge(head_norm, tail_norm):
                existing_relations = G[head_norm][tail_norm].get('relations', [])
                existing_relations.append({
                    'label': rel['label'],
                    'score': rel['score'],
                    'doc_id': doc_id,
                    'metadata': rel.get('metadata', {})
                })
                G[head_norm][tail_norm]['relations'] = existing_relations
            else:
                G.add_edge(head_norm, tail_norm,
                          relations=[{
                              'label': rel['label'],
                              'score': rel['score'],
                              'doc_id': doc_id,
                              'metadata': rel.get('metadata', {})
                          }])
        
        return G
    
    def find_entity_clusters(self, all_entities: Set[str]) -> List[Set[str]]:
        """
        Find clusters of similar entities using connected components.
        
        Args:
            all_entities: Set of all unique entities
            
        Returns:
            List of entity clusters (sets)
        """
        # Create similarity graph
        similarity_graph = nx.Graph()
        entities_list = list(all_entities)
        
        for i, entity1 in enumerate(entities_list):
            similarity_graph.add_node(entity1)
            for j, entity2 in enumerate(entities_list[i+1:], i+1):
                similarity = self.calculate_entity_similarity(entity1, entity2)
                if similarity >= self.similarity_threshold:
                    similarity_graph.add_edge(entity1, entity2, weight=similarity)
        
        # Find connected components (clusters)
        clusters = list(nx.connected_components(similarity_graph))
        return clusters
    
    def merge_graphs(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        """
        Merge multiple graphs by clustering similar entities.
        
        Args:
            graphs: List of NetworkX directed graphs
            
        Returns:
            Merged NetworkX directed graph
        """
        # Collect all entities
        all_entities = set()
        entity_to_graph = {}
        
        for i, graph in enumerate(graphs):
            for node in graph.nodes():
                all_entities.add(node)
                if node not in entity_to_graph:
                    entity_to_graph[node] = []
                entity_to_graph[node].append(i)
        
        # Find entity clusters
        clusters = self.find_entity_clusters(all_entities)
        
        # Create entity mapping from original to canonical
        entity_mapping = {}
        canonical_entities = {}
        
        for cluster in clusters:
            # Choose canonical entity (most frequent or longest name)
            cluster_list = list(cluster)
            canonical = max(cluster_list, key=lambda x: (len(entity_to_graph.get(x, [])), len(x)))
            
            # Initialize canonical entity with default values
            canonical_entities[canonical] = {
                'original_texts': set(),
                'entity_type': None,
                'doc_sources': set(),
                'metadata': set()
            }
            
            for entity in cluster:
                entity_mapping[entity] = canonical
                
                # Collect information from all graphs
                for graph_idx in entity_to_graph.get(entity, []):
                    graph = graphs[graph_idx]
                    if entity in graph.nodes():
                        node_data = graph.nodes[entity]
                        
                        # Safely handle original_text which might be missing or in different formats
                        original_text = node_data.get('original_text', entity)
                        if isinstance(original_text, str):
                            canonical_entities[canonical]['original_texts'].add(original_text)
                        elif isinstance(original_text, (list, set)):
                            for text in original_text:
                                if isinstance(text, str):
                                    canonical_entities[canonical]['original_texts'].add(text)
                        
                        # Update entity type if not set
                        if canonical_entities[canonical]['entity_type'] is None:
                            canonical_entities[canonical]['entity_type'] = node_data.get('entity_type')
                        
                        # Update document sources
                        if 'doc_sources' in node_data:
                            canonical_entities[canonical]['doc_sources'].update(node_data['doc_sources'])
                        
                        # Handle metadata
                        if 'metadata' in node_data:
                            metadata = node_data['metadata']
                            if isinstance(metadata, dict):
                                canonical_entities[canonical]['metadata'].add(tuple(metadata.items()))
                            elif metadata:  # Skip None or empty metadata
                                canonical_entities[canonical]['metadata'].add(metadata)
        
        # Create merged graph
        merged_graph = nx.DiGraph()
        
        # Add nodes with proper fallback values
        for canonical, info in canonical_entities.items():
            # Ensure all required attributes exist with proper defaults
            original_texts = info.get('original_texts', {canonical})
            entity_type = info.get('entity_type', 'UNKNOWN')
            doc_sources = info.get('doc_sources', set())
            
            # Convert metadata back to dictionary format with proper handling
            metadata_dicts = []
            for meta in info.get('metadata', set()):
                if isinstance(meta, tuple):
                    metadata_dicts.append(dict(meta))
                elif meta:  # Skip None or empty metadata
                    metadata_dicts.append(meta)
            
            # Combine metadata from all instances
            combined_metadata = {}
            for md in metadata_dicts:
                if md:
                    combined_metadata.update(md)
            
            merged_graph.add_node(canonical,
                                original_texts=original_texts,
                                entity_type=entity_type,
                                doc_sources=doc_sources,
                                metadata=combined_metadata)
        
        # Add edges
        for graph in graphs:
            for head, tail, edge_data in graph.edges(data=True):
                canonical_head = entity_mapping.get(head, head)
                canonical_tail = entity_mapping.get(tail, tail)
                
                if canonical_head in merged_graph and canonical_tail in merged_graph:
                    if merged_graph.has_edge(canonical_head, canonical_tail):
                        existing_relations = merged_graph[canonical_head][canonical_tail].get('relations', [])
                        existing_relations.extend(edge_data.get('relations', []))
                        merged_graph[canonical_head][canonical_tail]['relations'] = existing_relations
                    else:
                        merged_graph.add_edge(canonical_head, canonical_tail,
                                            relations=edge_data.get('relations', []))
        
        return merged_graph

    def process_graph_query(self, query: str) -> Tuple[str, nx.DiGraph, dict]:
        """
        Process query using graph-based RAG approach.
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (answer, merged_graph, document_texts)
        """
        # Initialize document texts
        document_texts = {
            'page_content_graph_1': '',
            'page_content_graph_2': '',
            'page_content_graph_3': '',
            'page_content_graph_4': ''
        }
        
        # Retrieve relevant documents
        print(f"Processing with Graph RAG for query: {query}")
        documents = self.retrieve_relevant_documents(query)
        
        # Store document texts
        for i, doc in enumerate(documents[:4]):
            col_name = f'page_content_graph_{i+1}'
            document_texts[col_name] = doc['text'][:10000] if 'text' in doc else ''
        
        if not documents:
            return "I couldn't find sufficient information to answer this question.", nx.DiGraph(), document_texts
        
        # Extract relations and build individual graphs
        all_graphs = []
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            relations = self.extract_relations_from_document(
                doc['text'],
                doc_id,
                doc.get('metadata', {})
            )
            
            if relations:
                graph = self.create_graph_from_relations(relations, doc_id)
                all_graphs.append(graph)
        
        # Merge graphs if we have multiple
        if len(all_graphs) > 1:
            merged_graph = self.merge_graphs(all_graphs)
        elif all_graphs:
            merged_graph = all_graphs[0]
        else:
            merged_graph = nx.DiGraph()
        
        # Generate context from merged graph
        sentences_with_sources = self.generate_sentences_from_merged_graph(merged_graph)
        
        # Combine with original document context
        context_parts = ["=== Knowledge Graph Context ==="]
        for item in sentences_with_sources:
            context_parts.append(f"- {item['sentence']}")
            if item['sources']:
                context_parts.append(f"  Sources: {', '.join(item['sources'])}")
        
        context_parts.append("\n=== Document Context ===")
        for doc in documents:
            text = doc.get('text', '')
            if len(text) > 1000:  # Smaller context for graph-based
                text = text[:1000] + "... [truncated]"
            
            source = doc.get('metadata', {}).get('source', 'unknown')
            context_parts.append(f"\nDocument from PMID {source}:")
            context_parts.append(text)
        
        context = "\n".join(context_parts)
        
        # Generate answer using graph-based chain
        answer = self.graph_llm_chain.run({
            "question": query,
            "context": context
        }).strip()
        
        return answer, merged_graph, document_texts

    def generate_sentences_from_merged_graph(self, merged_graph: nx.DiGraph) -> List[Dict]:
        sentences_with_sources = []
        
        for head, tail, edge_data in merged_graph.edges(data=True):
            relations = edge_data.get('relations', [])
            
            # Get node data with proper fallbacks
            head_data = merged_graph.nodes[head]
            tail_data = merged_graph.nodes[tail]
            
            # Safely get original_texts with fallback
            head_texts = head_data.get('original_texts', {head})
            tail_texts = tail_data.get('original_texts', {tail})
            
            # Choose display text (longest available)
            head_text = max(head_texts, key=len) if head_texts else head
            tail_text = max(tail_texts, key=len) if tail_texts else tail
            
            # Group relations by type
            relation_groups = defaultdict(list)
            for rel in relations:
                relation_groups[rel['label']].append(rel['score'])
            
            # Calculate geometric mean for each relation type
            relation_scores = {}
            for rel_type, scores in relation_groups.items():
                relation_scores[rel_type] = self.geometric_mean(scores)
            
            # Collect all unique sources
            sources = set()
            for rel in relations:
                if rel.get('metadata') and 'source' in rel['metadata']:
                    sources.add(rel['metadata']['source'])
            
            # Format sources as PubMed links
            formatted_sources = [f"https://pubmed.ncbi.nlm.nih.gov/{source}" for source in sources if source]
            
            # Generate sentence
            if len(relation_scores) == 1:
                rel_type, score = next(iter(relation_scores.items()))
                verb = self._get_relation_verb(rel_type)
                doc_count = len(set(rel['doc_id'] for rel in relations if rel.get('doc_id')))
                sentence = f"{head_text} {verb} {tail_text} (confidence: {score:.4f}, documents: {doc_count})."
            else:
                verbs = [self._get_relation_verb(rel_type) for rel_type in relation_scores.keys()]
                overall_score = self.geometric_mean(list(relation_scores.values()))
                doc_count = len(set(rel['doc_id'] for rel in relations if rel.get('doc_id')))
                
                if len(verbs) == 2:
                    sentence = f"{head_text} {verbs[0]} and {verbs[1]} {tail_text} (confidence: {overall_score:.4f}, documents: {doc_count})."
                else:
                    sentence = f"{head_text} {', '.join(verbs[:-1])}, and {verbs[-1]} {tail_text} (confidence: {overall_score:.4f}, documents: {doc_count})."
            
            sentences_with_sources.append({
                'sentence': sentence[0].upper() + sentence[1:],
                'sources': formatted_sources
            })
        
        return sentences_with_sources

    def _get_relation_verb(self, relation: str) -> str:
        """Get the appropriate verb for a relation type."""
        verb_map = {
            "INDIRECT-DOWNREGULATOR": "indirectly downregulates",
            "INDIRECT-UPREGULATOR": "indirectly upregulates",
            'DIRECT-REGULATOR': "directly regulates",
            'ACTIVATOR': "activates",
            'INHIBITOR': "inhibits",
            'AGONIST': "acts as an agonist for",
            'AGONIST-ACTIVATOR': "acts as an agonist activator for",
            'AGONIST-INHIBITOR': "acts as an agonist inhibitor for",
            'ANTAGONIST': "acts as an antagonist for",
            'PRODUCT-OF': "is produced by",
            'SUBSTRATE': "serves as a substrate for",
            'SUBSTRATE_PRODUCT-OF': "is both substrate and product of",
            'PART-OF': "is part of"
        }
        return verb_map.get(relation, relation.lower().replace('-', ' '))

    def geometric_mean(self, scores: List[float]) -> float:
        """Calculate geometric mean of scores."""
        if not scores:
            return 0.0
        if any(score <= 0 for score in scores):
            return 0.0
        return np.exp(np.mean(np.log(scores)))

    def process_query(self, query: str) -> Tuple[str, dict]:
        """
        Process a user query using either basic RAG or graph-based approach.
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (answer, document_texts)
        """
        if self.should_use_graph_method(query):
            answer, _, document_texts = self.process_graph_query(query)
        else:
            answer, document_texts = self.process_basic_rag_query(query)
        
        return answer, document_texts

    def visualize_merged_graph(self, merged_graph: nx.DiGraph, output_file: str = "merged_graph.html") -> str:
        """
        Visualize the merged graph using pyvis.
        
        Args:
            merged_graph: Merged NetworkX directed graph
            output_file: Output HTML file name
            
        Returns:
            Path to the generated HTML file
        """
        net = Network(
            notebook=True,
            height="800px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            cdn_resources='remote'
        )
        
        # Add nodes with proper fallbacks
        for node, data in merged_graph.nodes(data=True):
            # Safely get original_texts with fallback
            original_texts = data.get('original_texts', {node})
            if not original_texts:  # Handle empty sets
                original_texts = {node}
                
            entity_type = data.get('entity_type', 'UNKNOWN')
            doc_sources = data.get('doc_sources', set())
            
            # Choose display text
            display_text = max(original_texts, key=len) if original_texts else node
            
            color = 'lightblue' if entity_type == 'CHEMICAL' else 'lightgreen'
            title = f"Entity: {display_text} ({entity_type})\nSources: {len(doc_sources)} documents\nAliases: {', '.join(original_texts)}"
            
            net.add_node(node, label=display_text, color=color, title=title)
        
        # Rest of the method remains the same...
        # Add edges
        for head, tail, edge_data in merged_graph.edges(data=True):
            relations = edge_data.get('relations', [])
            
            # Group relations and calculate scores
            relation_groups = defaultdict(list)
            for rel in relations:
                relation_groups[rel['label']].append(rel['score'])
            
            relation_info = []
            for rel_type, scores in relation_groups.items():
                mean_score = self.geometric_mean(scores)
                relation_info.append(f"{rel_type}: {mean_score:.3f}")
            
            doc_count = len(set(rel['doc_id'] for rel in relations if rel.get('doc_id')))
            
            label = ', '.join(relation_groups.keys())
            title = f"Relations: {', '.join(relation_info)}\nDocuments: {doc_count}"
            
            net.add_edge(head, tail, label=label, title=title)
        
        net.show_buttons(filter_=['physics'])
        return net.show(output_file)

def process_csv_with_checkpoint(input_csv_path: str, output_csv_path: str, checkpoint_interval: int = 5):
    """
    Process questions from a CSV file and write answers with checkpointing.
    
    Args:
        input_csv_path: Path to input CSV file containing questions
        output_csv_path: Path to output CSV file with answers
        checkpoint_interval: Number of rows to process before saving a checkpoint
    """
    # Initialize the Smart RAG Graph Merger
    merger = SmartRAGGraphMerger(
        glirel_model_path="./Drugprot_REL_model",
        spacy_model_path="./NER_Model/model-best",
        similarity_threshold=0.8
    )
    
    # Check if we're resuming from a checkpoint
    checkpoint_path = output_csv_path + '.checkpoint'
    start_row = 0
    
    if os.path.exists(output_csv_path):
        # Load existing output to determine where to resume
        try:
            existing_df = pd.read_csv(output_csv_path)
            start_row = len(existing_df)
            print(f"Resuming from row {start_row}")
        except:
            pass
    
    # Read input CSV
    df = pd.read_csv(input_csv_path)
    total_rows = len(df)
    
    # Initialize output columns if they don't exist
    if 'GENERATED_ANSWER' not in df.columns:
        df['GENERATED_ANSWER'] = ''
    
    # Add document text columns if they don't exist
    for i in range(1, 5):
        col_name = f'page_content_graph_{i}'
        if col_name not in df.columns:
            df[col_name] = ''
    
    # Process rows with progress bar
    for i in tqdm(range(start_row, total_rows), initial=start_row, total=total_rows, desc="Processing questions"):
        try:
            question = df.iloc[i]['question']  # Assuming 'question' column exists
            if pd.isna(question) or not str(question).strip():
                df.at[i, 'GENERATED_ANSWER'] = "No question provided"
                continue
                
            # Process the question
            answer, document_texts = merger.process_query(str(question).strip())
            df.at[i, 'GENERATED_ANSWER'] = answer
            
            # Save document texts to their respective columns
            for col_name, text in document_texts.items():
                df.at[i, col_name] = text
            
            # Save checkpoint periodically
            if (i + 1) % checkpoint_interval == 0 or i == total_rows - 1:
                # Save both the main output and checkpoint
                df.to_csv(output_csv_path, index=False)
                df.to_csv(checkpoint_path, index=False)
                
        except Exception as e:
            print(f"\nError processing row {i}: {str(e)}")
            df.at[i, 'GENERATED_ANSWER'] = f"Error processing question: {str(e)}"
            
            # Initialize empty document texts in case of error
            for col_name in [f'page_content_graph_{i}' for i in range(1,5)]:
                if col_name not in df.columns:
                    df[col_name] = ''
            
            # Save error state to checkpoint
            df.to_csv(output_csv_path, index=False)
            df.to_csv(checkpoint_path, index=False)
            
            # Clear GPU memory and continue
            merger.clear_gpu_memory()
            continue
    
    # Final save and cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"\nProcessing complete. Results saved to {output_csv_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process questions from CSV and generate answers.")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--checkpoint', type=int, default=5, help='Checkpoint interval (rows)')
    
    args = parser.parse_args()
    
    process_csv_with_checkpoint(
        input_csv_path=args.input,
        output_csv_path=args.output,
        checkpoint_interval=args.checkpoint
    )

if __name__ == "__main__":
    main()
