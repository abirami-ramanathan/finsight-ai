import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class RAGGenerator:

    def __init__(
        self,
        model_name="google/flan-t5-base"
    ):

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            )
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(
            self.device
        )

    def generate(
        self,
        question,
        contexts,
        project_context="",
        max_new_tokens=180
    ):

        context_text = "\n\n".join(
            [
                (
                    f"Source: {item['source']}\n"
                    f"{item['text']}"
                )
                for item in contexts
            ]
        )

        prompt = f"""
You are a financial education assistant.

Answer the user's question using only the information
contained in the provided documents.

Question:
{question}

Documents:
{context_text}

Give a direct answer.

Explain the concept in simple language.

Use relevant details from the documents.

Do not copy unrelated text.

Do not invent information.

Do not give personalized investment advice.

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(
            self.device
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()