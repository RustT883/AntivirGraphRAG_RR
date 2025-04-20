from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    faithfulness,
    context_recall,
    context_precision,
)
import torch
import gc
from datasets import load_dataset

amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
eval_dataset = amnesty_qa["eval"].select(range(1,3))

llm = ChatOllama(model="qwen2.5:latest",verbose=False,timeout=600,num_ctx=8192,disable_streaming=False)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

result = evaluate(
    eval_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
    run_config =RunConfig(timeout=600, max_retries=20, max_wait=50,log_tenacity=False),
)

dataframe = result.to_pandas()
dataframe.to_csv("ragas_evaluation_test.csv")
