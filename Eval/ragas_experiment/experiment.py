from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_community.document_loaders import CSVLoader
from ragas import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import DirectoryLoader
import gc
import torch
import pandas as pd

torch.cuda.empty_cache()
gc.collect()

#torch_config
device = torch.device("cuda")

#load_csv
loader = CSVLoader(file_path='all_texts_for_drugs_remove.csv',
                   csv_args={'delimiter' : '\t',}, source_column="PubMedID")

data = loader.load()


run_config = RunConfig(max_retries = 50)

generator_llm = ChatOllama(
    model="qwen2.5:latest", 
    timeout=600, 
    temperature = 0.0,
    disable_streaming = False
)
generator_llm = LangchainLLMWrapper(generator_llm)

output_file = "sytnetic_qa_ragas.csv"
write_header = True

generator_embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
generator_embeddings = LangchainEmbeddingsWrapper(generator_embeddings)

goldens_antivir = pd.DataFrame()

generator = TestsetGenerator(
    llm = generator_llm,
    embedding_model=generator_embeddings
)

for document in data:
    try:
        testset = generator.generate_with_langchain_docs(
            documents = [document],
            testset_size=1
        )
        dataframe = testset.to_pandas()
        dataframe.to_csv(output_file, mode="a", index=False, header=write_header)
        write_header=False
    
    except ValueError as e:
        print(f"Skipping document due to error: {e}")
    
#goldens_antivir = pd.concat([goldens_antivir, dataframe])
#goldens_antivir = goldens_antivir.to_csv("sytnetic_qa_ragas.csv")
    



