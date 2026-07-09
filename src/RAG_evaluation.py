import json
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GeminiModel

from config import settings
from src.PrivacyComplianceRAG import PrivacyComplianceRAG


def run_deepeval_suite():
    settings.EVALUATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rag_system = PrivacyComplianceRAG()
    json_path = settings.EVALUATION_DATA_DIR / "evaluation_dataset.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Not found file with test: {json_path.resolve()}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        gold_standard = json.load(f)
    eval_model = GeminiModel(
        model=settings.EVALUATION_MODEL_NAME,
        temperature=0.0
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.6, model=eval_model),
        FaithfulnessMetric(threshold=0.6, model=eval_model)
    ]

    test_cases = []

    for idx, item in enumerate(gold_standard, 1):
        query = item["input"]
        print(f"[{idx}/{len(gold_standard)}] Processing: {item.get('id', 'TC')}")

        generated_answer, retrieved_contexts = rag_system.ask(query)

        test_case = LLMTestCase(
            input=query,
            actual_output=generated_answer,
            retrieval_context=retrieved_contexts,
            expected_output=item.get("expected_output")
        )
        test_cases.append(test_case)
    evaluate(test_cases=test_cases, metrics=metrics)


if __name__ == "__main__":
    run_deepeval_suite()