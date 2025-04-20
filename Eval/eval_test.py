import warnings
import chromadb
import torch
from deepeval.integrations import trace_llama_index
from deepeval.integrations.llama_index import DeepEvalFaithfulnessEvaluator
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

device = torch.device("cuda")
torch.cuda.empty_cache()

#llm = Ollama(model="llama2:latest")
#response = llm.complete("Who is Bruce Willis?")
#print(response)



warnings.filterwarnings("ignore", category=DeprecationWarning) 
callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
handler = StdOutCallbackHandler()

trace_llama_index(auto_eval=True)

llm = Ollama(model="mistral:latest", request_timeout=120.0)

Settings.llm = Ollama(model="mistral:latest", request_timeout=120.0)

model_name = '/home/rust/Downloads/IBMC/NLP/embed/nomic-embed-text-v1.5'

embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)
Settings.embed_model = embed_model


db = chromadb.PersistentClient(path="/home/rust/Downloads/IBMC/NLP/antiviral_rag/vectorstore_antiviral")
print(db.list_collections())
chroma_collection = db.get_collection(name="langchain")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model = embed_model
)

rag_application = index.as_query_engine()

user_input = "What is ritonavir?"

response_object = rag_application.query(user_input)

evaluator = DeepEvalFaithfulnessEvaluator()
evaluation_result = evaluator.evaluate_response(
    query=user_input, response=response_object
)
print(evaluation_result)


