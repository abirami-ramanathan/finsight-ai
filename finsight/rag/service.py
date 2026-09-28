from pathlib import Path

from .indexer import FAISSIndexer
from .generator import RAGGenerator
from .project_context import ProjectContext


class RAGService:

    def __init__(
        self,
        index_dir,
        top_k=5,
        root_dir=None
    ):
        self.indexer = FAISSIndexer.load(
            Path(index_dir)
        )

        self.generator = RAGGenerator()

        self.project_context = ProjectContext(
            root_dir or "."
        )

        self.top_k = top_k

    def classify_question(self, question):

        q = question.lower()

        module_a_keywords = [
            "technical indicator",
            "technical indicators",
            "features",
            "feature engineering",
            "input features",
            "indicators used",
            "rsi",
            "macd",
            "bollinger",
            "bollinger bands",
            "ema",
            "mfi",
            "stochastic",
            "volatility",
            "garch"
        ]

        module_b_keywords = [
            "forecasting results",
            "forecast result",
            "forecasting model",
            "forecast model",
            "prediction model",
            "prediction results",
            "performed well",
            "performed best",
            "best model",
            "which model",
            "better model",
            "best forecasting",
            "best prediction",
            "r2",
            "mape",
            "mae",
            "rmse",
            "forecast",
            "predict",
            "prediction"
        ]

        module_c_keywords = [
            "recommendation",
            "recommendations",
            "recommended",
            "knn",
            "linucb",
            "bandit",
            "contextual bandit",
            "risk score",
            "rerank",
            "re-rank",
            "exploration",
            "exploitation",
            "regret",
            "reward"
        ]

        if any(
            keyword in q
            for keyword in module_a_keywords
        ):
            return "A"

        if any(
            keyword in q
            for keyword in module_c_keywords
        ):
            return "C"

        if any(
            keyword in q
            for keyword in module_b_keywords
        ):
            return "B"

        return "RAG"

    def ask(self, question):

        module = self.classify_question(
            question
        )

        if module == "RAG":

            contexts = self.indexer.search(
                question,
                top_k=self.top_k
            )

            answer = self.generator.generate(
                question=question,
                contexts=contexts,
                project_context=""
            )

        else:

            contexts = []

            answer = self.project_context.get_context(
                question,
                module
            )

        sources = []

        source_map = {}

        for result in contexts:

            source = result["source"]

            if source not in source_map:

                source_map[source] = (
                    len(source_map) + 1
                )

                sources.append(
                    {
                        "id": source_map[source],
                        "source": source,
                        "chunk_id": result["chunk_id"],
                        "distance": result["distance"]
                    }
                )

        citation_text = ""

        if sources:

            citation_text = "\n\nSources:\n"

            for source in sources:

                citation_text += (
                    f"[{source['id']}] "
                    f"{source['source']}\n"
                )

        return {
            "question": question,
            "module": module,
            "answer": answer,
            "answer_with_citations": (
                answer + citation_text
            ),
            "citations": sources,
            "retrieved_context": contexts
        }