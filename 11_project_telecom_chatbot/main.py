"""
CLI entry point for the telecom RAG chatbot.
Usage: python main.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
from rag_chain import build_chain
from groq import APIError, RateLimitError

load_dotenv()


def main():
    print("=== Telecom Customer Care Chatbot (RAG) ===")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    try:
        chain = build_chain()
    except (ValueError, OSError) as exc:
        print(f"Setup error: {exc}")
        return

    while True:
        question = input("Customer: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        try:
            for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
        except RateLimitError:
            print("Groq rate limit reached. Wait a minute before trying again.")
        except APIError as exc:
            print(f"Groq request failed: {exc}")
        print("\n")


if __name__ == "__main__":
    main()
