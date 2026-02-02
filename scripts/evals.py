"""RAG evaluation: run the same agent as the app on an eval dataset and score vs ground truth.

Flow:
  1. Agent (agent.py) is a RAG agent: it uses an embedding model + document database and
     only answers from those documents (search → retrieve → answer).
  2. Eval dataset has N questions (e.g. 20) with ground-truth answers derived from the
     same documents.
  3. For each question we run the same agent: it searches the documents and returns an
     answer.
  4. Eval metrics compare each answer to the ground truth (correctness, exact match,
     contains).

We run the eval agent on each eval dataset question, collect the response, then
MLflow evaluates that response against the expected answer from the dataset.

Dataset schema (per record):
  - inputs: {"question": "Where are travelers required to check-in when travelling for OSCORP?"}
  - expectations: {"expected_response": "They may only use the Shadow Terminal...", "expected_document": "travel.md"}
"""

import asyncio
import logging
import uuid

import mlflow
from mlflow.genai import evaluate
from mlflow.genai.datasets import search_datasets
from mlflow.genai.scorers import Correctness, scorer

from demo_mlflow_agent_tracing.agent import build_agent
from demo_mlflow_agent_tracing.base import ContextSchema
from demo_mlflow_agent_tracing.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

@scorer(name="ground_truth_exact_match", description="1.0 if model output exactly matches expected_response (case-insensitive, stripped), else 0.0")
def ground_truth_exact_match(outputs: str, expectations: dict) -> float:
    """Score 1.0 when output exactly matches ground truth, else 0.0."""
    expected = expectations.get("expected_response")
    if expected is None:
        return 0.0
    return 1.0 if (outputs or "").strip().lower() == str(expected).strip().lower() else 0.0


@scorer(name="ground_truth_contains", description="1.0 if expected_response appears in model output (case-insensitive), else 0.0")
def ground_truth_contains(outputs: str, expectations: dict) -> float:
    """Score 1.0 when ground truth is contained in the output, else 0.0."""
    expected = expectations.get("expected_response")
    if expected is None:
        return 0.0
    return 1.0 if str(expected).strip().lower() in (outputs or "").strip().lower() else 0.0


@scorer(name="expected_document_mentioned", description="1.0 if expected_document (e.g. travel.md) appears in model output, else 0.0")
def expected_document_mentioned(outputs: str, expectations: dict) -> float:
    """Score 1.0 when the expected source document is mentioned in the output."""
    expected_doc = expectations.get("expected_document")
    if not expected_doc:
        return 1.0  # no expectation = pass
    out = (outputs or "").strip().lower()
    # Match document name with or without extension (e.g. travel.md or travel)
    doc_name = str(expected_doc).strip().lower().removesuffix(".md")
    return 1.0 if doc_name in out or expected_doc.strip().lower() in out else 0.0


def _row_to_inputs_expectations(row) -> tuple[dict, dict]:
    """Extract inputs and expectations from a dataset row.

    Dataset schema we support:
      - inputs: {"question": "..."}
      - expectations: {"expected_response": "...", "expected_document": "travel.md"}
    Handles both dict columns and flattened columns (e.g. inputs.question).
    """
    inputs = row.get("inputs")
    expectations = row.get("expectations")
    if not isinstance(inputs, dict):
        # Flattened columns, e.g. row["inputs.question"]
        question = row.get("inputs.question", row.get("question", ""))
        inputs = {"question": question} if question else {}
    if not isinstance(expectations, dict):
        exp_resp = row.get("expectations.expected_response", row.get("expected_response"))
        exp_doc = row.get("expectations.expected_document", row.get("expected_document"))
        expectations = {}
        if exp_resp is not None:
            expectations["expected_response"] = exp_resp
        if exp_doc is not None:
            expectations["expected_document"] = exp_doc
    return inputs or {}, expectations or {}


def _content_to_str(content: str | list) -> str:
    """Normalize token content to string (content can be str or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", block) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content) if content else ""


async def run_agent(agent, question: str) -> str:
    """Run the RAG agent on one question: it searches the document DB and returns the answer."""
    messages = [{"role": "user", "content": question}]
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    context = ContextSchema(user_info="eval")
    input_state = {"messages": messages, "user_info": "eval"}

    final_content: list[str] = []
    async for token, metadata in agent.astream(
        input=input_state,
        config=config,
        context=context,
        stream_mode="messages",
    ):
        if token.content:
            final_content.append(_content_to_str(token.content))

    return "".join(final_content) if final_content else ""


def main() -> None:
    """Run RAG eval: for each question in the dataset, run the agent (search docs → answer), then score vs ground truth."""
    settings = Settings()
    if settings.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    if settings.MLFLOW_EXPERIMENT_NAME:
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    mlflow.autolog(disable=True)

    dataset_name = settings.EVAL_DATASET_NAME or "oscorp_policies_validation_set"
    datasets = search_datasets(
        filter_string=f"name = '{dataset_name}'",
        max_results=1,
    )
    if not datasets:
        raise SystemExit(
            f"Dataset '{dataset_name}' not found. Create and upload it first with: "
            "uv run python scripts/generate_eval_dataset.py"
        )
    dataset = datasets[0]
    df = dataset.to_df()
    num_questions = len(df)
    logger.info("Using dataset: %s (id=%s), %s questions", dataset.name, dataset.dataset_id, num_questions)

    async def run_eval() -> None:
        # Same RAG agent as the app: embedding model + document DB + search tool + LLM
        model = f"openai:/{settings.OPENAI_MODEL_NAME}"
        agent = await build_agent()
        scorers = [
            Correctness(model=model),  # uses expectations.expected_response
            ground_truth_exact_match,               # uses expectations.expected_response
            ground_truth_contains,                  # uses expectations.expected_response
            expected_document_mentioned,            # uses expectations.expected_document
        ]

        # Dataset schema: inputs = {"question": "..."}, expectations = {"expected_response": "...", "expected_document": "travel.md"}
        # For each question: run agent (searches documents, returns answer), then we score vs ground truth
        eval_data = []
        for idx, row in df.iterrows():
            inputs, expectations = _row_to_inputs_expectations(row)
            question = inputs.get("question", "")
            try:
                output = await run_agent(agent, question)
            except Exception as e:
                logger.exception("Prediction failed for question=%r: %s", question[:80], e)
                raise
            eval_data.append({"inputs": inputs, "outputs": output, "expectations": expectations})
            logger.info("Prediction %s/%s done", len(eval_data), num_questions)

        try:
            results = evaluate(
                data=eval_data,
                scorers=scorers,
            )
            logger.info("Eval metrics: %s", results.metrics)
            if "eval_results_table" in results.tables:
                logger.info("Eval results table:\n%s", results.tables["eval_results_table"].to_string())
        except Exception as e:
            err = str(e).lower()
            if any(
                x in err
                for x in ("503", "service unavailable", "application is not available", "not serving")
            ):
                raise SystemExit(
                    "The model API at OPENAI_BASE_URL returned 503 (service unavailable). "
                    "Ensure the model server is running and reachable, then re-run the eval."
                ) from e
            raise

    asyncio.run(run_eval())


if __name__ == "__main__":
    main()
