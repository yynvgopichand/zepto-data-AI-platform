import sys

sys.path.append("src")

from retriever import Retriever

retriever = Retriever(top_k=3)

results = retriever.search(
    "How long do I have to report damaged items?"
)

for result in results:
    print(result["filename"])
    print(result["distance"])
    print()