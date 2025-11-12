# AntivirGraphRAG

**Note**: This is **not** related to Microsoft's GraphRAG approach. This is an independent project for antiviral drug research.

## About

AntivirGraphRAG is a knowledge graph-based retrieval augmented generation system for antiviral drug research. This project combines Named Entity Recognition (NER) and Relation Extraction (RE) models to build a comprehensive knowledge graph from biomedical literature. This project utilizes GLiREL.

## Quick Start

### Prerequisites

- Python 3.10+
- Virtual environment tool (venv or conda)

### Installation

1. **Create and activate a virtual environment:**

   Using venv:
   ```bash
   python -m venv antivirgraphrag-env
   source antivirgraphrag-env/bin/activate  # On Windows: antivirgraphrag-env\Scripts\activate
   ```

   Or using conda:
   ```bash
   conda create -n antivirgraphrag-env python=3.10
   conda activate antivirgraphrag-env
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Data & Model Setup

3. **Download required files from Google Drive:**

   Download the following files from [Google Drive](https://drive.google.com/drive/folders/1-ve0xTQXbxggwnByFfTu8AhSG2xMkCgc?usp=sharing):
   - `all_texts_for_drugs_processed.csv`
   - `NER_Model` (folder)
   - `Drugprot_REL_model` (folder)

   Place these in your project root directory.

### Initialization

4. **Create local vector stores:**
   ```bash
   python create_bm25s_index.py
   python make_chroma_store.py
   ```

### Usage

5. **Run the main application:**
   ```bash
   python antivir_graphrag.py
   ```
## Models

- **NER Model**: Trained for extracting drug-related entities from biomedical text
- **RE Model**: Trained for extracting relations between entities (Drugprot_REL_model)

