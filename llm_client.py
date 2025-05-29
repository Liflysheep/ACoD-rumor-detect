import os
from openai import BadRequestError
from anthropic import Anthropic
from openai import OpenAI


class LLMClient:
    def __init__(self, base_url: str = None, api_key: str = None):
        self.openai_client = OpenAI(base_url=base_url, api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY")
        self.anthropic_client = Anthropic()

    def request(
        self,
        payload: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        if model.startswith("claude"):
            message = self.anthropic_client.messages.create(
                messages=[{"role": "user", "content": payload}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = message.content[0].text
            token_count = message.usage.output_tokens
        else:
            try:
                completion = self.openai_client.chat.completions.create(
                    messages=[{"role": "user", "content": payload}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = completion.choices[0].message.content
                token_count = completion.usage.completion_tokens
            except BadRequestError as e:
                if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 400:
                    # Handle content inspection failure
                    response = "此新闻不能被检测"
                    token_count = 5
                else:
                    # Re-raise other BadRequestErrors
                    raise

        return response, token_count


if __name__ == "__main__":
    llm = LLMClient()
    response, count = llm.request("hello", "claude-3-7-sonnet-latest")
    print(response, count)