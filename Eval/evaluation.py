import json
import os
import chromadb
import pandas as pd
from pydantic import BaseModel
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import pipeline
from deepeval.synthesizer import Synthesizer
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from deepeval.synthesizer.config import FiltrationConfig
from typing import Optional, Tuple, List, Union, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import trimAndLoadJson
from langchain_community.callbacks import get_openai_callback
from deepeval.models.gpt_model import CustomChatOpenAI
from deepeval.synthesizer.config import ContextConstructionConfig

device = torch.device("cuda")

#deepeval version is 2.3.8
#previous version of chromadb was 0.5.23
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    quantization_config=quantization_config # remove to load FP16 model
)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3"
)


class CustomMistral7B(DeepEvalBaseLLM):
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            num_return_sequences=1,
            max_new_tokens=512,
            do_sample=True,
            top_k=5,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if schema is not None:
            # Create parser required for JSON confinement using lmformatenforcer
            parser = JsonSchemaParser(schema.model_json_schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )
            # Output and load valid JSON
            output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
            output = output_dict[0]["generated_text"][len(prompt):]
            json_result = json.loads(output)
            # Return valid JSON object according to the schema DeepEval supplied
            return schema(**json_result)

        # Generate response using the chat template
        response = pipeline(prompt)[0]["generated_text"]

        # Extract the assistant's response from the generated text
        # assistant_response = response[len(chat_template):].strip()

        # Return valid JSON object according to the schema DeepEval supplied
        return response

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Mistral-7B v0.3"

#torch_config
device = torch.device("cuda")

#embedder = "nomic-embed-text-v1.5"


model_local = CustomMistral7B()

filtration_config = FiltrationConfig(
    synthetic_input_quality_threshold=0.5,
    max_quality_retries=1,
    critic_model=model_local
)

context_construction_config = ContextConstructionConfig(
    max_contexts_per_document = 3,
    max_context_length = 3,
    chunk_size = 1024,
    chunk_overlap = 0,
    context_quality_threshold = 0.5,
    context_similarity_threshold = 0.5,
    max_retries = 3
)

goldens_antivir = pd.DataFrame()
synthesizer = Synthesizer(model=model_local, filtration_config=filtration_config)

#antivir_df = pd.read_csv("all_texts_for_drugs_remove.csv", sep="\t")
#antivir_documents = antivir_df["Text"].tolist()


# Loop through the list and create a new file for each string
#for i, string in enumerate(antivir_documents[:10]):
    # Generate a unique file name for each string
#    file_name = f'./antivir_docs/antivir_doc_{i + 1}.txt'
    
    # Open the file in write mode and save the string
#    with open(file_name, 'w') as file:
#        file.write(string)

#print("All strings have been successfully saved to separate files.")

#directory_path = "./antivir_documents"

#texts = [file for file in os.listdir(directory_path) if file.endswith('.txt') and #os.path.isfile(os.path.join(directory_path, file))]
#paths = []

#for text in range(len(texts)):
#    a = f"./antivir_docs/{texts[text]}"
#    paths.append(a)

try:
    synthesizer.generate_goldens_from_docs(
        document_paths=["./antivir_documents/text_list.txt"],
        context_construction_config=context_construction_config
    )
    dataframe = synthesizer.to_pandas()
    dataframe.to_csv("goldens_antivir_2.csv")
#    goldens_antivir = pd.concat([dataframe, goldens_antivir])
#    goldens_antivir = goldens_antivir.to_csv("goldens_antivir_500_chunk_size.csv")
except json.decoder.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    print("Skipping Invalid JSON") 

                 
       

#for row in range(len(antivir_rows)):
#    try:
#        text = text_splitter.split_text(antivir_rows[row])
#        synthesizer.generate_goldens_from_contexts(
#            contexts = [text],
#            max_goldens_per_context=1
#        )




