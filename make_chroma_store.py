from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
import os
import torch
import gc

#torch_config
device = torch.device("cuda")
#torch.cuda.set_per_process_memory_fraction(0.8)

#load_csv
loader = CSVLoader(file_path='all_texts_for_drugs_processed.csv',
                   csv_args={'delimiter' : '\t',}, source_column="PubMedID")

data = loader.load()

#split_csv
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600, #the character length of the chunk
    chunk_overlap = 60, #the character length of the overlap between chunks
    separators=[".", " ", ","],
    length_function = len, #the length function - character length
)

splits = text_splitter.transform_documents(data)

# Use OllamaEmbeddings instead of HuggingFaceEmbeddings
core_embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text",
    model_kwargs={'device': 'cuda'}
)

persist_directory = './vectorstore_antiviral_chunk_size_600/'

def split_list(splits, chunk_size):
    for i in range(0, len(splits), chunk_size):
        yield splits[i:i + chunk_size]
        
split_docs_chunked = split_list(splits, 41000)

for split_docs_chunk in split_docs_chunked:
    vectordb = Chroma.from_documents(
        documents=split_docs_chunk,
        embedding=core_embeddings_model,
        persist_directory=persist_directory,
    )
    torch.cuda.empty_cache()
    gc.collect()
