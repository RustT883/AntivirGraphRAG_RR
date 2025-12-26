from langchain_community.retrievers import BM25SRetriever
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import psutil
import os
from pprint import pprint

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"Memory usage: {mem:.2f} MB")


print("Loading documents...")
print_memory_usage()

loader = CSVLoader(
        file_path="all_texts_for_drugs_processed.csv",
        source_column="PubMedID",
        metadata_columns=["Drugs"],
        csv_args={
            'delimiter': '\t',
            'quotechar': '"',
            }
)

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=[".", " ", ","],
        length_function = len,
)

split_docs = []
for doc in tqdm(docs, desc="Splitting documents"):
    split_docs.extend(text_splitter.split_documents([doc]))
    if len(split_docs) % 100 == 0:
        print_memory_usage()


print(f"Page content: {split_docs[0].page_content[:100]} \n\n Metadata: {split_docs[0].metadata}")

text_contents = []
metadata = []

for doc in tqdm(split_docs, desc="Extracting contents"):
    text_contents.append(doc.page_content)
    metadata.append(doc.metadata)

print("Creating BM25 retriever...")

retriever = BM25SRetriever.from_texts(
        texts=text_contents,
        metadatas=metadata,
        k=100,
        persist_directory='bm25_antivir'
)

print("Process completed successfully!")

