"""RAG evaluation: run the same agent as the app on an eval dataset and score vs ground truth.

Uses MLflow's evaluate() with predict_fn so that:
  - The agent is invoked per row and returns the full state (messages, tool calls).
  - Scorers receive structured outputs and can evaluate retrieval (did the right doc get
    retrieved?) and answer quality (Correctness, Completeness, Relevance).

Dataset schema (per record):
  - inputs: {"question": "Where are travelers required to check-in when travelling for OSCORP?"}
  - expectations: {"expected_response": "...", "expected_document": "travel.md"}
"""

import asyncio
import logging
from typing import Any
from uuid import uuid4

import mlflow
from demo_mlflow_agent_tracing.agent import build_agent, format_config, format_context, format_input
from demo_mlflow_agent_tracing.mcp_server import SearchResult
from demo_mlflow_agent_tracing.settings import Settings
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai import evaluate
from mlflow.genai.scorers import Completeness, Correctness, RelevanceToQuery, scorer

mlflow.langchain.autolog(run_tracer_inline=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def get_messages(outputs: dict[str, Any]) -> list[BaseMessage]:
    """Get messages from agent outputs."""
    return outputs.get("messages", [])


def get_tool_calls(outputs: dict[str, Any]) -> list[tuple[dict[str, Any], ToolMessage]]:
    """Parse tool call and response pairs from outputs."""
    messages = get_messages(outputs)
    ai_messages: list[AIMessage] = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages: list[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]
    tool_calls: list[dict[str, Any]] = sum(
        [m.tool_calls for m in ai_messages if m.tool_calls], start=[]
    )
    pairs: list[tuple[dict[str, Any], ToolMessage]] = []
    for tc in tool_calls:
        tid = tc.get("id")
        resp = next((m for m in tool_messages if m.tool_call_id == tid), None)
        if resp is not None:
            pairs.append((tc, resp))
    return pairs


def get_retrieved_documents(outputs: dict[str, Any]) -> list[str]:
    """Parse retrieved document names from tool responses."""
    pairs = get_tool_calls(outputs)
    names: list[str] = []
    for _tc, response in pairs:
        raw = getattr(response, "artifact", None) or {}
        if isinstance(raw, dict):
            structured = raw.get("structured_content", {})
        else:
            structured = {}
        if not structured:
            continue
        try:
            search_result = SearchResult.model_validate(structured)
            for doc in search_result.documents:
                names.append(doc.metadata.get("file", ""))
        except Exception:
            pass
    return names


@scorer(name="Retrieval")
def retrieval_score(outputs: dict[str, Any], expectations: dict[str, Any]) -> Feedback:
    """Check if the expected document was retrieved during the conversation."""
    expected_document = expectations.get("expected_document")
    if expected_document is None:
        return Feedback(value="yes", rationale="No expected document provided")
    try:
        retrieved = get_retrieved_documents(outputs)
        if expected_document in retrieved:
            return Feedback(value="yes", rationale="Expected document was retrieved by tool calls")
        return Feedback(value="no", rationale="Expected document was not retrieved by tool calls")
    except Exception as e:
        logger.error("Error parsing outputs for retrieval score: %s", e)
        return Feedback(
            value="no",
            rationale=f"There was an error parsing the outputs: {e!s}",
            error=e,
        )


async def run_agent(question: str) -> dict[str, Any]:
    """Run the agent on one question; returns full state (messages, etc.) for scorers."""
    user = "evals"
    input_state = format_input(content=question, user_identifier=user)
    config = format_config(thread_id=str(uuid4()))
    context = format_context(user_identifier=user)
    agent = await build_agent()
    try:
        response = await agent.ainvoke(input=input_state, config=config, context=context)
        return response
    except Exception as e:
        logger.exception("Agent invocation failed for question=%r: %s", question[:80], e)
        return {"status": "error", "message": str(e)}


def predict(question: str) -> dict[str, Any]:
    """Sync predict_fn for MLflow evaluate: called as predict(question=...) from inputs dict."""
    return asyncio.run(run_agent(question or ""))


def main() -> None:
    """Run RAG eval using MLflow evaluate with predict_fn and full-state scorers."""
    load_dotenv()
    settings = Settings()
    if settings.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    if settings.MLFLOW_EXPERIMENT_NAME:
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    dataset_name = settings.EVAL_DATASET_NAME or "oscorp_policies_validation_set"
    client = MlflowClient()
    matched = client.search_datasets(
        filter_string=f"name LIKE '{dataset_name}'",
        max_results=5,
    )
    if not matched:
        raise SystemExit(
            f"Dataset '{dataset_name}' not found. Create and upload it first with: "
            "uv run python scripts/generate_eval_dataset.py"
        )
    dataset = matched[0]
    logger.info("Using dataset: %s", dataset.name)

    model = f"openai:/{settings.OPENAI_MODEL_NAME}"
    scorers_list = [
        Correctness(model=model),
        Completeness(name="Completeness", model=model),
        RelevanceToQuery(name="Relevance", model=model),
        retrieval_score,
    ]

    try:
        results = evaluate(data=dataset, scorers=scorers_list, predict_fn=predict)
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


if __name__ == "__main__":
    main()
