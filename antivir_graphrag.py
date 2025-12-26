# RAG Graph Merger with LLM Integration for CHEMICAL-GENE Relations (Single-Pass Version)
import pandas as pd
import spacy
import scispacy
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
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
import webbrowser
import os
import warnings
import logging

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*INFO:.*")
warnings.filterwarnings("ignore")

class RAGGraphMergerWithLLM:
    def __init__(self, glirel_model_path: str, spacy_model_path: str, similarity_threshold: float = 0.8):
        """
        Initialize the RAG Graph Merger with LLM integration.
        
        Args:
            glirel_model_path: Path to trained GLiREL model
            spacy_model_path: Path to trained spaCy NER model
            similarity_threshold: Threshold for merging similar entities (default: 0.8)
        """
        self.model = GLiREL.from_pretrained(glirel_model_path, map_location='cuda')
        self.nlp = spacy.load(spacy_model_path)
        self.similarity_threshold = similarity_threshold
        
        # Load medembed-large model
        self.embedding_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')
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
    
    def initialize_retriever(self):
        """Initialize the document retrieval system."""
        core_embeddings_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'device': "cpu", 'trust_remote_code': True},
            encode_kwargs={'batch_size': 1, 'normalize_embeddings': True}
        )
        
        persist_directory = "./vectorstore_antiviral_chunk_size_600"
        self.vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=core_embeddings_model
        )
        
        vectorstore_retriever = self.vectordb.as_retriever(search_kwargs={'k': 100})
        
        compressor = FlashrankRerank(top_n=15)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore_retriever
        )
    
    def initialize_llm(self):
        """Initialize the LLM for question answering."""
        self.llm = ChatOllama(
            base_url="http://localhost:11434",
            model="myaniu/qwen2.5-1m:7b",
            temperature=0.0,
            num_ctx=4096
        )
        
        self.prompt_template = """You are an expert in medicinal chemistry and pharmacology. 
        Below is context information extracted from scientific literature about chemical-gene interactions:
        
        {context}
        
        Based on this information, answer the following question. Follow these rules:
        1. Be precise and factual, only using information from the provided context
        2. If you don't know the answer, say "I couldn't find sufficient information to answer this question"
        3. For chemical-gene interactions, specify the type of relationship (e.g., activates, inhibits)
        4. Always include the PubMed source links when available, formatted as: https://pubmed.ncbi.nlm.nih.gov/[PMID]
        
        Question: {question}
        Answer: """
        
        self.qa_prompt = PromptTemplate.from_template(self.prompt_template)
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
    
    def retrieve_relevant_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for a query with their metadata."""
        docs = self.retriever.get_relevant_documents(query)
        return [{
            'text': doc.page_content,
            'metadata': doc.metadata
        } for doc in docs]
    
    def normalize_entity(self, entity_text: str) -> str:
        """Normalize entity text for consistent matching."""
        if isinstance(entity_text, list):
            entity_text = ' '.join(entity_text)
        
        # Handle None or empty strings
        if not entity_text or not isinstance(entity_text, str):
            return ""
        
        return entity_text.strip().lower()

    def calculate_entity_similarity(self, entity1: str, entity2: str) -> float:
        """
        Calculate similarity between two entities using multiple methods.
        Fixed version with robust tensor dimension handling.
        """
        def safe_truncate(text, max_words=100):  # Reduced from 512 to 100
            if not text or not isinstance(text, str):
                return ""
            words = text.split()[:max_words]
            return ' '.join(words)
        
        def safe_normalize(entity_text):
            if isinstance(entity_text, list):
                entity_text = ' '.join(entity_text)
            
            if not entity_text or not isinstance(entity_text, str):
                return ""
            
            return entity_text.strip().lower()
        
        # Normalize and truncate entities
        e1_norm = safe_normalize(entity1)
        e2_norm = safe_normalize(entity2)
        
        # Further truncate very long strings
        e1_norm = safe_truncate(e1_norm)
        e2_norm = safe_truncate(e2_norm)

        # Check for empty strings after normalization
        if not e1_norm or not e2_norm:
            return 0.0

        # Exact match
        if e1_norm == e2_norm:
            return 1.0
        
        # String similarity
        string_sim = SequenceMatcher(None, e1_norm, e2_norm).ratio()
        substring_sim = 0.9 if (e1_norm in e2_norm or e2_norm in e1_norm) else 0.0

        # Initialize cosine similarity
        cosine_sim = 0.0
        
        # Only proceed with embeddings if we have valid text
        try:
            # Additional safety check for very short or problematic texts
            if len(e1_norm) < 2 or len(e2_norm) < 2:
                return max(string_sim, substring_sim)
                
            with torch.no_grad():
                # Get embeddings - use numpy arrays for safer handling
                e1_embed = self.embedding_model.encode(
                    [e1_norm],  # Pass as list to ensure batch dimension
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=1
                )
                e2_embed = self.embedding_model.encode(
                    [e2_norm],  # Pass as list to ensure batch dimension
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=1
                )
                
                # Ensure we have 2D arrays (batch_size, embedding_dim)
                if e1_embed.ndim == 1:
                    e1_embed = e1_embed.reshape(1, -1)
                if e2_embed.ndim == 1:
                    e2_embed = e2_embed.reshape(1, -1)
                
                # Verify we have valid arrays with same dimensions
                if e1_embed.shape[1] == e2_embed.shape[1] and e1_embed.shape[1] > 0:
                    # Calculate cosine similarity directly on numpy arrays
                    cosine_sim = cosine_similarity(e1_embed, e2_embed)[0][0]
                else:
                    print(f"Embedding dimension mismatch: {e1_embed.shape} vs {e2_embed.shape}")
                    cosine_sim = 0.0
                    
        except Exception as e:
            print(f"Error in embedding calculation: {str(e)}")
            # Fall back to string-based similarity
            cosine_sim = string_sim

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
    
    def extract_relations_from_document(self, text: str, doc_id: str = None, metadata: dict = None) -> List[Dict]:
        doc = self.nlp(text)
        tokens = [token.text for token in doc]

        # Early guards (prevents empty-span crashes)
        if not tokens or len(tokens) < 2 or not doc.ents:
            return []

        # IMPORTANT: pass only span indices (start, end)
        # Use inclusive end as you were doing, but now it's just [start, end]
        ner_spans = [[ent.start, ent.end - 1] for ent in doc.ents]

        labels_and_constraints = self.labels["glirel_labels"]
        labels_list = list(labels_and_constraints.keys())

        relations = self.model.predict_relations(
            tokens,
            labels_list,
            threshold=0.0,
            ner=ner_spans,
            top_k=3
        )

        relations = constrain_relations_by_entity_type(doc.ents, labels_and_constraints, relations)
        
        filtered_relations = []
        for rel in relations:
            if rel['score'] > 0.5:
                head_text = ' '.join(rel['head_text']) if isinstance(rel['head_text'], list) else rel['head_text']
                tail_text = ' '.join(rel['tail_text']) if isinstance(rel['tail_text'], list) else rel['tail_text']
                
                head_ent = next((ent for ent in doc.ents if ent.text == head_text), None)
                tail_ent = next((ent for ent in doc.ents if ent.text == tail_text), None)
                
                if (head_ent and tail_ent and
                    head_ent.label_ == "CHEMICAL" and
                    tail_ent.label_ == "GENE"):
                    rel['doc_id'] = doc_id
                    rel['metadata'] = metadata
                    filtered_relations.append(rel)
        
        return filtered_relations
    
    def create_graph_from_relations(self, relations: List[Dict], doc_id: str = None) -> nx.DiGraph:
        """Create a NetworkX graph from relations."""
        G = nx.DiGraph()
        
        for rel in relations:
            head_text = ' '.join(rel['head_text']) if isinstance(rel['head_text'], list) else rel['head_text']
            tail_text = ' '.join(rel['tail_text']) if isinstance(rel['tail_text'], list) else rel['tail_text']
            
            head_norm = self.normalize_entity(head_text)
            tail_norm = self.normalize_entity(tail_text)
            
            metadata = rel.get('metadata', {})
            G.add_node(head_norm,
                      original_text=head_text,
                      entity_type='CHEMICAL',
                      doc_sources={doc_id} if doc_id else set(),
                      metadata=metadata)
            G.add_node(tail_norm,
                      original_text=tail_text,
                      entity_type='GENE',
                      doc_sources={doc_id} if doc_id else set(),
                      metadata=metadata)
            
            if G.has_edge(head_norm, tail_norm):
                existing_relations = G[head_norm][tail_norm].get('relations', [])
                existing_relations.append({
                    'label': rel['label'],
                    'score': rel['score'],
                    'doc_id': doc_id,
                    'metadata': metadata
                })
                G[head_norm][tail_norm]['relations'] = existing_relations
            else:
                G.add_edge(head_norm, tail_norm,
                          relations=[{
                              'label': rel['label'],
                              'score': rel['score'],
                              'doc_id': doc_id,
                              'metadata': metadata
                          }])
        
        return G
    
    def find_entity_clusters(self, all_entities: Set[str]) -> List[Set[str]]:
        """Find clusters of similar entities using connected components."""
        similarity_graph = nx.Graph()
        entities_list = list(all_entities)
        
        for i, entity1 in enumerate(entities_list):
            similarity_graph.add_node(entity1)
            for j, entity2 in enumerate(entities_list[i+1:], i+1):
                similarity = self.calculate_entity_similarity(entity1, entity2)
                if similarity >= self.similarity_threshold:
                    similarity_graph.add_edge(entity1, entity2, weight=similarity)
        
        return list(nx.connected_components(similarity_graph))
    
    def merge_graphs(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        """Merge multiple graphs by clustering similar entities."""
        all_entities = set()
        entity_to_graph = {}
        
        for i, graph in enumerate(graphs):
            for node in graph.nodes():
                all_entities.add(node)
                if node not in entity_to_graph:
                    entity_to_graph[node] = []
                entity_to_graph[node].append(i)
        
        clusters = self.find_entity_clusters(all_entities)
        
        entity_mapping = {}
        canonical_entities = {}
        
        for cluster in clusters:
            cluster_list = list(cluster)
            canonical = max(cluster_list, key=lambda x: (
                len(entity_to_graph.get(x, [])),
                len(x)
            ))
            
            canonical_entities[canonical] = {
                'original_texts': set(),
                'entity_type': None,
                'doc_sources': set(),
                'metadata': set()
            }
            
            for entity in cluster:
                entity_mapping[entity] = canonical
                
                for graph_idx in entity_to_graph.get(entity, []):
                    graph = graphs[graph_idx]
                    if entity in graph.nodes():
                        node_data = graph.nodes[entity]
                        canonical_entities[canonical]['original_texts'].add(
                            node_data.get('original_text', entity)
                        )
                        canonical_entities[canonical]['entity_type'] = node_data.get('entity_type')
                        canonical_entities[canonical]['doc_sources'].update(
                            node_data.get('doc_sources', set())
                        )
                        if 'metadata' in node_data:
                            canonical_entities[canonical]['metadata'].add(
                                tuple(node_data['metadata'].items()) if isinstance(node_data['metadata'], dict) else node_data['metadata']
                            )
        
        merged_graph = nx.DiGraph()
        
        for canonical, info in canonical_entities.items():
            metadata_dicts = []
            for meta in info['metadata']:
                if isinstance(meta, tuple):
                    metadata_dicts.append(dict(meta))
                elif meta:
                    metadata_dicts.append(meta)
            
            combined_metadata = {}
            for md in metadata_dicts:
                if md:
                    combined_metadata.update(md)
            
            merged_graph.add_node(canonical,
                                 original_texts=info['original_texts'],
                                 entity_type=info['entity_type'],
                                 doc_sources=info['doc_sources'],
                                 metadata=combined_metadata)
        
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
    
    def geometric_mean(self, scores: List[float]) -> float:
        """Calculate geometric mean of scores."""
        if not scores:
            return 0.0
        if any(score <= 0 for score in scores):
            return 0.0
        return np.exp(np.mean(np.log(scores)))
    
    def generate_sentences_from_merged_graph(self, merged_graph: nx.DiGraph) -> List[Dict]:
        """Generate natural language sentences from the merged graph with sources."""
        sentences_with_sources = []
        
        for head, tail, edge_data in merged_graph.edges(data=True):
            relations = edge_data.get('relations', [])
            
            head_data = merged_graph.nodes[head]
            tail_data = merged_graph.nodes[tail]
            
            head_text = max(head_data['original_texts'], key=len) if head_data['original_texts'] else head
            tail_text = max(tail_data['original_texts'], key=len) if tail_data['original_texts'] else tail
            
            relation_groups = defaultdict(list)
            for rel in relations:
                relation_groups[rel['label']].append(rel['score'])
            
            relation_scores = {}
            for rel_type, scores in relation_groups.items():
                relation_scores[rel_type] = self.geometric_mean(scores)
            
            sources = set()
            for rel in relations:
                if rel.get('metadata') and 'source' in rel['metadata']:
                    sources.add(rel['metadata']['source'])
            
            formatted_sources = [f"https://pubmed.ncbi.nlm.nih.gov/{source}" for source in sources if source]
            
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
    
    def visualize_merged_graph(self, merged_graph: nx.DiGraph, output_file: str = "merged_graph.html") -> str:
        """Visualize the merged graph using pyvis."""
        net = Network(
            notebook=True,
            height="800px",
            width="100%",
            bgcolor="#FFFFFF",
            font_color="black",
            cdn_resources='remote'
        )
        
        for node, data in merged_graph.nodes(data=True):
            original_texts = data.get('original_texts', {node})
            entity_type = data.get('entity_type', 'UNKNOWN')
            doc_sources = data.get('doc_sources', set())
            
            display_text = max(original_texts, key=len) if original_texts else node
            
            color = 'lightblue' if entity_type == 'CHEMICAL' else 'lightgreen'
            title = f"Entity: {display_text} ({entity_type})\nSources: {len(doc_sources)} documents\nAliases: {', '.join(original_texts)}"
            
            net.add_node(node, label=display_text, color=color, title=title)
        
        for head, tail, edge_data in merged_graph.edges(data=True):
            relations = edge_data.get('relations', [])
            
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
    
    def answer_question_with_graph(self, query: str) -> Dict:
        """
        Process a user query and generate an answer with graph in a single pass.
        
        Args:
            query: The user's question/query
            
        Returns:
            Dictionary containing answer, graph visualization, and extracted relations
        """
        # Retrieve relevant documents
        documents = self.retrieve_relevant_documents(query)
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Extract relations and build graphs
        all_graphs = []
        all_relations = []
        
        for i, (doc_data, doc_id) in enumerate(zip(documents, doc_ids)):
            relations = self.extract_relations_from_document(
                doc_data['text'],
                doc_id,
                doc_data['metadata']
            )
            
            if relations:
                graph = self.create_graph_from_relations(relations, doc_id)
                all_graphs.append(graph)
                all_relations.extend(relations)
        
        # Merge graphs if we have any
        if all_graphs:
            merged_graph = self.merge_graphs(all_graphs)
            sentences_with_sources = self.generate_sentences_from_merged_graph(merged_graph)
            graph_html_path = self.visualize_merged_graph(merged_graph)
        else:
            merged_graph = nx.DiGraph()
            sentences_with_sources = []
            graph_html_path = None
        
        # Prepare context for LLM
        context_parts = ["=== Extracted Chemical-Gene Relations ==="]
        
        if sentences_with_sources:
            for item in sentences_with_sources:
                context_parts.append(f"- {item['sentence']}")
                if item['sources']:
                    context_parts.append(f"  Sources: {', '.join(item['sources'])}")
        else:
            context_parts.append("No specific chemical-gene relations found.")
        
        # Add original document excerpts
        context_parts.append("\n=== Supporting Document Excerpts ===")
        for doc in documents:
            text = doc['text']
            if len(text) > 1000:
                text = text[:1000] + "... [truncated]"
            
            source = doc['metadata'].get('source', 'unknown')
            context_parts.append(f"\nDocument from PMID {source}:")
            context_parts.append(text)
        
        context = "\n".join(context_parts)
        
        # Get answer from LLM
        answer = self.llm_chain.run({
            "question": query,
            "context": context
        }).strip()
        
        return {
            "answer": answer,
            "graph_html": graph_html_path,
            "relations": all_relations,
            "documents": documents,
            "merged_graph": merged_graph
        }

import traceback
def chat_interface():
    """Interactive chat interface for querying chemical-gene relations."""
    merger = RAGGraphMergerWithLLM(
        glirel_model_path="./Drugprot_REL_model",
        spacy_model_path="./NER_Model/model-best",
        similarity_threshold=0.8
    )
    
    print("Chemical-Gene Relation Knowledge Graph Chat")
    print("Type your question or 'exit' to quit\n")
    
    while True:
        query = input("Question: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        
        try:
            # Process the query and get answer in one pass
            result = merger.answer_question_with_graph(query)

            # Display the answer
            print("\nAnswer:")
            print(result["answer"])

            # Display the graph if available
            if result["graph_html"]:
                print(f"\nGraph visualization saved to: {result['graph_html']}")

            # Option to show extracted relations
            show_relations = input("\nShow extracted relations? (y/n): ").strip().lower()
            if show_relations == 'y' and result["relations"]:
                print("\nExtracted Relations:")
                for i, rel in enumerate(result["relations"], 1):
                    head = ' '.join(rel['head_text']) if isinstance(rel['head_text'], list) else rel['head_text']
                    tail = ' '.join(rel['tail_text']) if isinstance(rel['tail_text'], list) else rel['tail_text']
                    print(f"{i}. {head} --[{rel['label']} (score: {rel['score']:.2f})]-> {tail}")

            print("\n" + "="*50 + "\n")

        except Exception as e:
            print(f"Error processing query: {e}")
            traceback.print_exc()   # <-- THIS prints the full stack trace + exact failing line
            continue

if __name__ == "__main__":
    chat_interface()
