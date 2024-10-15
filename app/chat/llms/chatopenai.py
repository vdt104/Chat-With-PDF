from langchain_openai import ChatOpenAI
import os

# def build_llm(chat_args, model):
#     return ChatOpenAI(
#         model=model,
#         base_url="https://models.inference.ai.azure.com",
#         api_key=os.environ.get("GITHUB_API_KEY")
#     )


    # return ChatOpenAI(
    #     model="gpt-4o-mini",
    #     base_url="https://models.inference.ai.azure.com",
    #     api_key=os.environ.get("GITHUB_API_KEY")
    # )

    # return ChatOpenAI(
    #     model="gpt-4o",
    #     base_url="https://models.inference.ai.azure.com",
    #     api_key=os.environ.get("GITHUB_API_KEY")
    # )

    # return ChatOpenAI(
    #     model='Meta-Llama-3.1-405B-Instruct',
    #     base_url="https://api.sambanova.ai/v1",
    #     api_key=os.environ.get("SAMBANOVA_API_KEY"),
    # )

def build_llm(chat_args, model):
    if model == 'Meta-Llama-3.1-405B-Instruct':
        return ChatOpenAI(
            model=model,
            base_url="https://api.sambanova.ai/v1",
            api_key=os.environ.get("SAMBANOVA_API_KEY")
        )
    else:
        return ChatOpenAI(
            model=model,
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ.get("GITHUB_API_KEY")
        )
