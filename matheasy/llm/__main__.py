"""Check that your API key and the pinned model work: python -m matheasy.llm"""

import sys

from . import LLMClient


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "What is 1/2 + 1/3? Answer with the fraction only."
    llm = LLMClient(run_id="smoke-test")
    session = llm.session("smoke", condition="smoke", budget_tokens=None)
    result = session.call("smoke", [{"role": "user", "content": prompt}])
    print(f"provider: {llm.backend.provider}  model: {llm.config.model()}  config: {llm.config.hash}")
    print(f"reply:    {result.text.strip()}")
    print(f"tokens:   in {result.input_tokens}, out {result.output_tokens}, thinking {result.thought_tokens}"
          f"{'  (cached)' if result.cached else ''}")
    print(f"log:      {llm.log_path}")


if __name__ == "__main__":
    main()
