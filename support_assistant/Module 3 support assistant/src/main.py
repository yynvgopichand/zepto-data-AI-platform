from fastapi import FastAPI

from .graph import graph
from .models import AskRequest, QAResponse


app = FastAPI(
    title="Zepto GenAI Service",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Zepto GenAI Service is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/ask", response_model=QAResponse)
def ask_question(request: AskRequest):
    result = graph.invoke(
        {
            "query": request.query
        }
    )

    return QAResponse.model_validate(
        result["response"]
    )