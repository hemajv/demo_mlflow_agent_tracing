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

logger = logging.getLogger(__name__)


def get_messages(outputs: dict[str, Any]) -> list[BaseMessage]:
    """Get messages from outputs."""
    messages: list[BaseMessage] = outputs.get("messages", [])
    return messages


def get_tool_calls(outputs: dict[str, Any]) -> list[tuple[dict[str, Any], ToolMessage]]:
    """Parse tool call and response pairs from outputs."""
    # Split messages into AI and Tool messages
    messages = get_messages(outputs)
    ai_messages: list[AIMessage] = [message for message in messages if isinstance(message, AIMessage)]
    tool_messages: list[ToolMessage] = [message for message in messages if isinstance(message, ToolMessage)]

    # Parse tool calls from AI messages
    tool_calls: list[dict[str, Any]] = sum([message.tool_calls for message in ai_messages if message.tool_calls], start=[])

    # Pair tools calls with their responses
    tool_call_pairs: list[tuple[dict[str, Any], ToolMessage]] = []
    for tool_call in tool_calls:
        tool_call_id = tool_call.get("id")
        tool_response = [message for message in tool_messages if message.tool_call_id == tool_call_id][0]
        tool_call_pairs.append((tool_call, tool_response))

    return tool_call_pairs


def get_retrived_documents(outputs: dict[str, Any]):
    """Parse out retrieved documents."""
    # Get tool responses
    tool_call_pairs = get_tool_calls(outputs=outputs)
    tool_responses = [pair[1] for pair in tool_call_pairs]

    # Get document names
    structured_tool_responses = [response.artifact.get("structured_content", {}) for response in tool_responses]
    search_results = [SearchResult.model_validate(response) for response in structured_tool_responses if response]
    retrieved_documents = sum([search_result.documents for search_result in search_results], start=[])
    retrieved_document_names = [doc.metadata.get("file", "") for doc in retrieved_documents]

    return retrieved_document_names


# Define custom scorers
@scorer(name="Retrieval")
def retrieval_score(outputs: dict[str, Any], expectations: dict[str, Any]):
    """Check if the expected document was retrieved during the conversation."""
    expected_document = expectations.get("expected_document")
    if expected_document is None:
        return Feedback(value="yes", rationale="No expected document provided")

    try:
        # Get retrieved documents
        retrieved_document_names = get_retrived_documents(outputs=outputs)

        # Check if expected document is in retrieved documents
        if expected_document in retrieved_document_names:
            return Feedback(value="yes", rationale="Expected document was retrieved by tool calls")
        return Feedback(value="no", rationale="Expected document was not retrieved by tool calls")
    except Exception as e:
        logger.error(e)
        return Feedback(value="no", rationale=f"There was an error parsing the outputs: {str(e)}", error=e)


# @scorer
# def tool_calling_score(outputs, expectations):
#     pass


# @scorer
# def num_turns_score(outputs, expectations):
#     pass


async def run_agent(question: str) -> dict[str, Any]:
    """Run the agent."""
    # Construct the graph input
    user = "evals"
    input = format_input(content=question, user_identifier=user)
    config = format_config(thread_id=str(uuid4()))
    context = format_context(user_identifier=user)

    # Build the agent
    agent = await build_agent()

    # Get the response from the agent
    try:
        response = await agent.ainvoke(input=input, config=config, context=context)
        return response
    except Exception as e:
        return {"status": "error", "message": str(e)}


def predict(question: str):
    """Get a prediction from the agent."""
    result = asyncio.run(run_agent(question=question))
    return result


def main():
    """Run eval process."""
    # Load settings
    load_dotenv()
    settings = Settings()
    if settings.MLFLOW_TRACKING_URI is not None:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    if settings.MLFLOW_EXPERIMENT_NAME is not None:
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # Fetch dataset
    dataset_name = "oscorp_policies_validation_set"
    client = MlflowClient()
    matched_datasets = client.search_datasets(filter_string=f"name LIKE '{dataset_name}'", max_results=5)
    if len(matched_datasets) == 0:
        raise ValueError(f"No dataset matching '{dataset_name}' found")
    dataset = matched_datasets[0]

    # Collect scorers
    model = f"openai:/{settings.OPENAI_MODEL_NAME}"
    scorers = [
        Correctness(model=model),
        # Guidelines(model=model, name="support_quality", guidelines="Response must be helpful, accurate, and professional"),
        Completeness(name="Completeness", model=model),
        RelevanceToQuery(name="Relevance", model=model),
        retrieval_score,
        # tool_calling_score,
        # num_turns_score,
    ]

    # Run evaluation
    results = evaluate(data=dataset, scorers=scorers, predict_fn=predict)

    # Print results
    logger.info("Evaluation results")
    logger.info(f"{results.metrics}")

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()