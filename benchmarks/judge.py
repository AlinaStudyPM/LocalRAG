# benchmarks/judge.py
import os
import asyncio

from openai import OpenAI, AsyncOpenAI
from ragas.llms import llm_factory


OLLAMA_BASE_URL = "http://localhost:11434/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SBER_CHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GENAPI_BASE_URL = "https://proxy.gen-api.ru/v1"


def get_judge_ollama(model: str, temperature: float = 0.0) -> object:
    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return llm_factory(
        model=model,
        client=client,
        adapter="instructor",
        temperature=temperature,
    )


def get_judge_openrouter(model: str, temperature: float = 0.0):
    api_key = os.getenv("OPENROUTER_KEY", "")
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return llm_factory(
        model=model,
        client=client,
        adapter="instructor",
        temperature=temperature,
    )

def get_judge_sber(model: str, temperature: float = 0.0):
    api_key =os.getenv("GIGACHAT_KEY", "")
    client = AsyncOpenAI(
        base_url=SBER_CHAT_BASE_URL,
        api_key=api_key,
    )
    return llm_factory(
        model=model,
        client=client,
        temperature=temperature,
    )

def get_judge_genapi(model: str, temperature: float = 0.0):
    api_key = os.getenv("GENAPI_KEY", "")
    client = AsyncOpenAI(
        base_url=GENAPI_BASE_URL,
        api_key=api_key,
    )
    return llm_factory(
        model=model,
        client=client,
        temperature=temperature,
        max_tokens=4096,
    )

async def _test_judge(judge, name: str, client: AsyncOpenAI):
    """Пробный запрос: дёшево, быстро, информативно."""
    try:
        # Если client обёрнут instructor'ом, берём базовый AsyncOpenAI
        openai_client = client
        if hasattr(client, "client") and isinstance(client.client, AsyncOpenAI):
            openai_client = client.client

        response = await openai_client.chat.completions.create(
            model=getattr(judge, "model", "unknown"),
            messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
            max_tokens=5,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        print(f"   {name}: OK (ответ: '{content}')")
    except Exception as e:
        print(f"   {name}: ОШИБКА — {e}")


async def main():
    print("Проверка judge.py")

    j = get_judge_ollama("llama3.2:latest")
    print(f"   Judge тип: {type(j).__name__}")

    j_or = get_judge_openrouter("openai/gpt-4o-mini")
    print(f"   Judge тип: {type(j_or).__name__}")
    #await _test_judge(j_or, "OR", j_or.client)

    j_genapi = get_judge_genapi("claude-4")
    print(f"   Judge тип: {type(j_genapi).__name__}")
    await _test_judge(j_genapi, "GenAPI", j_genapi.client)

    j_sber = get_judge_sber("GigaChat")
    print(f"   Judge тип (Sber): {type(j_sber).__name__}")
    #await _test_judge(j_sber, "Sber", j_sber.client)

if __name__ == "__main__":
    asyncio.run(main())
