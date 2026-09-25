
from typing import Literal

from langgraph.graph import StateGraph, START, END

from .config import MOCK_LLM
from .models import GraphState, QAResponse
from .prompts import build_rag_prompt
from .rag import retrieve_chunks


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_with_mock(query: str) -> str:
    query_lower = query.lower()

    if any(
        keyword in query_lower
        for keyword in POLICY_KEYWORDS
    ):
        return "policy_question"

    return "general_question"


def classify_intent(state: GraphState):
    query = state["query"]

    if MOCK_LLM:
        intent = classify_with_mock(query)
    else:
        # Optional real-LLM extension.
        # Implement an LLM classification call here.
        raise NotImplementedError(
            "Real LLM classification is optional "
            "and has not been configured."
        )

    return {
        "intent": intent
    }


def mock_retrieval_answer(
    query: str,
    chunks: list[dict],
) -> QAResponse:
    if not chunks:
        return QAResponse(
            answer=(
                "I could not find relevant information "
                "in the Zepto policy documents."
            ),
            sources=[],
            confidence=0.0,
        )

    top_chunk_snippet = chunks[0]["text"][:200]

    return QAResponse(
        answer=(
            "Based on the retrieved context: "
            f"{top_chunk_snippet}"
        ),
        sources=[
            chunk["id"]
            for chunk in chunks
        ],
        confidence=1.0,
    )


def real_retrieval_answer(
    query: str,
    chunks: list[dict],
) -> QAResponse:
    prompt = build_rag_prompt(query, chunks)

    # Optional extension:
    # Call your selected LLM provider here.
    # Parse and validate its result with QAResponse.
    raise NotImplementedError(
        "Configure a real LLM provider for MOCK_LLM=0."
    )


def retrieve_and_answer(state: GraphState):
    query = state["query"]

    # Retrieval is always performed.
    chunks = retrieve_chunks(
        query=query,
        top_k=3,
    )

    if MOCK_LLM:
        response = mock_retrieval_answer(
            query,
            chunks,
        )
    else:
        response = real_retrieval_answer(
            query,
            chunks,
        )

    return {
        "retrieved_chunks": chunks,
        "response": response.model_dump(),
    }


def direct_answer(state: GraphState):
    query = state["query"]

    if MOCK_LLM:
        response = QAResponse(
            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            sources=[],
            confidence=1.0,
        )
    else:
        # Optional extension:
        # Call the real LLM without retrieval.
        raise NotImplementedError(
            "Configure a real LLM provider for MOCK_LLM=0."
        )

    return {
        "response": response.model_dump()
    }


def route_after_classification(
    state: GraphState,
) -> Literal[
    "retrieve_and_answer",
    "direct_answer",
]:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node(
        "classify_intent",
        classify_intent,
    )

    workflow.add_node(
        "retrieve_and_answer",
        retrieve_and_answer,
    )

    workflow.add_node(
        "direct_answer",
        direct_answer,
    )

    workflow.add_edge(
        START,
        "classify_intent",
    )

    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge(
        "retrieve_and_answer",
        END,
    )

    workflow.add_edge(
        "direct_answer",
        END,
    )

    return workflow.compile()


graph = build_graph()