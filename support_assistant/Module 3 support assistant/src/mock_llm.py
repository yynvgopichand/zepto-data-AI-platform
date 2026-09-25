
from models import QAResponse, Source


def generate_mock_answer(
    query: str,
    retrieved_documents: list[dict],
) -> QAResponse:
    query_lower = query.lower()

    if not retrieved_documents:
        return QAResponse(
            answer="I could not find relevant information in the Zepto policy documents.",
            sources=[],
            confidence="low",
        )

    # Choose the first retrieved document as the primary context.
    primary = retrieved_documents[0]

    answer = primary["text"]
    filename = primary["filename"]

    # Use a simple deterministic relevance score.
    relevance = 1.0 / (1.0 + primary["distance"])

    if relevance >= 0.7:
        confidence = "high"
    elif relevance >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return QAResponse(
        answer=answer,
        sources=[
            Source(
                document=filename,
                relevance=round(relevance, 4),
            )
        ],
        confidence=confidence,
    )