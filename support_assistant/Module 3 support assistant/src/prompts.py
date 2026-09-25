
def build_rag_prompt(
    query: str,
    retrieved_chunks: list[dict],
) -> str:
    context = "\n\n".join(
        [
            f"[{chunk['id']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    return f"""
ROLE:
You are a helpful Zepto policy assistant.

CONTEXT:
Use only the following Zepto policy documents:
{context}

TASK:
Answer the user's question using the provided context.

FORMAT:
Return a concise answer in JSON-compatible form.
Do not include information outside the provided context.
If the context does not contain the answer, say that
the information is not available in the provided context.

LENGTH:
Keep the answer concise, preferably 1–3 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information not present in the
provided context. Do not invent policies, fees, or
delivery conditions.

FEW-SHOT EXAMPLE:
Question: What is the delivery fee for orders below INR 149?
Context: Standard delivery is free on orders over INR 149;
orders below this threshold incur a flat INR 25 delivery fee.
Answer: Orders below INR 149 incur a flat INR 25 delivery fee.

USER QUESTION:
{query}
"""