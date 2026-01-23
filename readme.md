# AntivirGraphRAG

> **⚠️ Important Note**: This project is **not** related to Microsoft's GraphRAG approach. This is an independent research project focused on antiviral drug discovery.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A knowledge graph-based Retrieval-Augmented Generation (RAG) system for antiviral drug research that combines Named Entity Recognition (NER) and Relation Extraction (RE) models to build comprehensive knowledge graphs from biomedical literature.

## Table of Contents
- [Overview](#-overview)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Data & Model Setup](#-data--model-setup)
- [Usage](#-usage)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Hardware Specifications](#-hardware-specifications)
- [Dependencies](#-dependencies)

## Overview

AntivirGraphRAG integrates:
- **Named Entity Recognition (NER)** for extracting drug-related entities
- **Relation Extraction (RE)** for identifying relationships between entities
- **Knowledge Graph Construction** for structured biomedical knowledge representation
- **Retrieval-Augmented Generation** for context-aware question answering

Built on the [GLiREL](https://github.com/urchade/GLiREL) framework.

## System Requirements

### Tested Hardware Configuration
- **GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM) + NVIDIA GeForce RTX 3060 Movile (6 GB VRAM) 
- **RAM**: 11th Gen Intel i7-11700 (8 cores and 16 threads, 32 GB) + AMD Ryzen 5600H (6 cores and 12 threads, 8 GB)
- **Storage**: 20GB free space for models and data
- **Peak and latency during inference**: <5.5 GB VRAM  with ~10 seconds per query end-to-end on full system

### Software
- **OS**: Tested on Linux (Ubuntu 22.04, Ubuntu 20.04)
- **Python**: 3.10 or higher
- **CUDA**: 12.8 (compatible with driver version 570.153.02)
- **NVIDIA Drivers**: 570.153.02

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/AntivirGraphRAG.git
cd AntivirGraphRAG
```

### 2. Create Virtual Environment

#### Option A: Using venv
```bash
python -m venv antivirgraphrag-env
source antivirgraphrag-env/bin/activate  # On Linux/Mac
# On Windows: antivirgraphrag-env\Scripts\activate
```

#### Option B: Using conda
```bash
conda create -n antivirgraphrag-env python=3.10
conda activate antivirgraphrag-env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Data & Model Setup

### Download Required Files
Download from [Zenodo](https://doi.org/10.5281/zenodo.18047501):
1. **Data**: `all_texts_for_drugs_processed.csv`
2. **NER Model**: `NER_Model.tar.gz`
3. **RE Model**: `Drugprot_REL_model.tar.gz`
4. **Verification**: `checksums.sha256`

Place all downloaded files in the project root directory.

### Initialize Vector Stores
```bash
# Create BM25 indices
python create_bm25s_index.py

# Create Chroma vector store
python make_chroma_store.py
```

## Usage

### Run Main Application
```bash
python antivir_graphrag.py
```

## Evaluation

### 1. Prepare Evaluation Data
```bash
# Download MedMCQA dataset (follow instructions in their repository)
# https://github.com/medmcqa/medmcqa

# Create antiviral subset
python create_medmcqa_antiviral_subset.py \
    --input ./path/to/medmcqa/train.json \
    --dict ./path/to/antivirals.csv
```

### 2. Run Ablation Studies
```bash
python antivir_graphrag_val_abl.py \
    --input_csv medmcqa_antiviral_drug_only_train_deduped.csv \
    --output_csv ablation_results.csv \
    --ablation "FULL_SYSTEM_K=10,NO_GRAPH_AUGMENTATION_K=10,DENSE_ONLY_RETRIEVAL_K=10,BM25_ONLY_RETRIEVAL_K=10,NO_ENTITY_MERGING_K=10,EDGES_OFF_K=10,FULL_SYSTEM_K=5" \
    --seed 0
```

### 3. Run Final Evaluation
```bash
python antivir_graphrag_val_abl.py \
    --eval_stats \
    --output_csv ablation_results.csv \
    --baseline "BM25_ONLY_RETRIEVAL_K=10"
```

## Project Structure
```
AntivirGraphRAG/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── antivir_graphrag.py       # Main application
├── antivir_graphrag_val_abl.py  # Evaluation script
├── create_bm25s_index.py    # BM25 index creation
├── make_chroma_store.py     # Chroma vector store creation
├── create_medmcqa_antiviral_subset.py  # Data preprocessing
├── Data/                    # Data directory
│   ├── antiviral_dictionary.csv
│   └── medmcqa_antiviral_drug_only_train_deduped.csv
├── Models/                  # Model directory
│   ├── NER_Model/
│   └── Drugprot_REL_model/
└── Results/                 # Evaluation results
```

## Hardware Specifications

### GPU-Intensive Dependencies
The following packages utilize GPU acceleration:
- **torch** (2.7.0+cu128): Core deep learning framework with CUDA 12.8 support
- **flash_attn** (2.8.3): Optimized attention mechanisms
- **transformers** (4.56.1): Hugging Face transformer models
- **nvidia-cuda-***: CUDA toolkit components (12.8 compatible)
- **onnxruntime** (1.22.0): Inference optimization

## Dependencies

### Core Libraries
- **GLiREL** (1.2.1)
- **LangChain** (0.3.25): LLM application framework
- **spaCy** (3.7.5): NLP processing
- **ChromaDB** (0.6.3): Vector database
- **sentence-transformers** (3.0.1): Embedding models

## Checksums

Checksums provided for all files. Zenodo models listed separately.

Generate: `python checksums.py`  
Verify: `python verify.py`

### GPU/CUDA Libraries
```txt
torch==2.7.0+cu128
torchvision==0.22.0+cu128
torchaudio==2.7.0+cu128
nvidia-cuda-runtime-cu12==12.8.57
nvidia-cudnn-cu12==9.7.1.26
flash_attn==2.8.3
```

### Full dependency list available in `requirements.txt`

