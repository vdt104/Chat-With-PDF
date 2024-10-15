from functools import partial
from .chatopenai import build_llm

llm_map = {
    "gpt-4o": partial(build_llm, model="gpt-4o"),
    "gpt-4o-mini": partial(build_llm, model="gpt-4o-mini"),
    "Meta-Llama-3.1-405B-Instruct": partial(build_llm, model="Meta-Llama-3.1-405B-Instruct"),
}
