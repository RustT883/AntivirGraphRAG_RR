from typing import Optional, Tuple, List, Union, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel
from langchain_core.outputs import ChatResult
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import trimAndLoadJson
from langchain_community.callbacks import get_openai_callback
from deepeval.synthesizer import Synthesizer

class CustomChatOpenAI(ChatOpenAI):
    format: str = None

    def __init__(self, format: str = None, **kwargs):
        super().__init__(**kwargs)
        self.format = format

    async def _acreate(
        self, messages: List[BaseMessage], **kwargs
    ) -> ChatResult:
        if self.format:
            kwargs["format"] = self.format
        return await super()._acreate(messages, **kwargs)

class OllamaLLM(DeepEvalBaseLLM):
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434/v1/",
        json_mode: bool = True,
        temperature: float = 0,
        *args,
        **kwargs
    ):

        self.model_name = model_name
        self.base_url = base_url
        self.json_mode = json_mode
        self.temperature = temperature
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self) -> CustomChatOpenAI:
        """Load and configure the Ollama model."""
        return CustomChatOpenAI(
            model_name=self.model_name,
            openai_api_key="ollama",  
            base_url=self.base_url,
            format="json" if self.json_mode else None,
            temperature=self.temperature,
            *self.args,
            **self.kwargs,
        )

    def generate(self, prompt: str, schema: Any = None) -> Union[Any, Tuple[str, float]]:

        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            if schema is not None:
                try:
                    # Try to parse the response using the schema
                    data = trimAndLoadJson(res.content, None)
                    return schema(**data)
                except Exception:
                    # If schema parsing fails, return parsed JSON
                    return trimAndLoadJson(res.content, None)
            return res.content, 0.0

    async def a_generate(self, prompt: str, schema: Any = None) -> Union[Any, Tuple[str, float]]:

        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
            if schema is not None:
                try:
                    # Try to parse the response using the schema
                    data = trimAndLoadJson(res.content, None)
                    return schema(**data)
                except Exception:
                    # If schema parsing fails, return parsed JSON
                    return trimAndLoadJson(res.content, None)
            return res.content, 0.0

    def get_model_name(self) -> str:
        """Get the name of the current model."""
        return self.model_name

    @property
    def __class__(self):
        from deepeval.models import GPTModel  
        return GPTModel

model_name="qwen2.5:latest"
model_local = OllamaLLM(model_name=model_name)

synthesizer = Synthesizer(model=model_local)
synthesizer.generate_goldens_from_contexts(
    contexts=[
        ["The Earth revolves around the Sun.", "Planets are celestial bodies."],
        ["Water freezes at 0 degrees Celsius.", "The chemical formula for water is H2O."],
    ]
)

print(res.content)

