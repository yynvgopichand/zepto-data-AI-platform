# Zepto Data & AI Platform Capstone Project

A three-module Data & AI capstone project covering data engineering, analytics and machine learning, and a policy-document-grounded GenAI/RAG service.

**Program:** Masai AIML - IIT Patna

---

## Project Overview

This repository contains three connected technical modules:

1. **Module 1 - Data Pipeline**

   * Scrapes book catalog data from Books to Scrape.
   * Cleans and transforms the data using pandas.
   * Converts GBP prices to INR using the fixed rate of **105.50 INR per GBP**.
   * Loads the data into a normalized SQLite database.
   * Executes SQL queries and reproduces a SQL JOIN using pandas.

2. **Module 2 - Analytics & Machine Learning**

   * Performs exploratory data analysis on the Titanic dataset.
   * Handles missing values and preprocessing.
   * Performs classification modeling and class-imbalance comparison.
   * Tunes a Random Forest model using grid search.
   * Performs multivariate linear regression.
   * Saves visualizations and the trained preprocessing/model pipeline.

3. **Module 3 - Zepto GenAI Support Assistant**

   * Implements a document-grounded RAG service for Zepto policy documents.
   * Uses local sentence-transformer embeddings and ChromaDB.
   * Uses LangGraph for workflow orchestration.
   * Provides structured Pydantic responses.
   * Exposes the service through FastAPI.
   * Supports Docker.
   * Includes a deterministic offline mock LLM mode for the graded baseline.

---

## Repository Structure

```text
Zepto Capstone Project/
|
+-- data_pipeline/
|   +-- Module 1 data pipeline/
|       +-- data_pipeline/
|           +-- pipeline.py
|           +-- README.md
|           +-- requirements.txt
|           +-- queries.sql
|           +-- books.db
|           +-- books_scraped.csv
|           +-- cleaned_books.csv
|           +-- join_comparison.csv
|           +-- query_outputs/
|
+-- analytics/
|   +-- Module 2 analytics/
|       +-- analytics_pipeline.py
|       +-- README.md
|       +-- titanic.csv
|       +-- models/
|       +-- outputs/
|
+-- support_assistant/
|   +-- Module 3 support assistant/
|       +-- src/
|       +-- docs/
|       +-- data/
|       +-- README.md
|       +-- requirements.txt
|       +-- Dockerfile
|       +-- test_retrieval.py
|
+-- .gitignore
+-- README.md
```

---

# Module 1 - Data Pipeline

## Objective

Build a raw-to-relational data pipeline using the public **Books to Scrape** website.

## Pipeline

The pipeline performs:

1. Web scraping using `requests` and `BeautifulSoup`.
2. Extraction of title, price, rating, availability, category, and URL.
3. Data cleaning using pandas.
4. GBP-to-INR conversion using the fixed rate of **105.50**.
5. Creation of a normalized SQLite database.
6. Storage of categories and books in separate relational tables.
7. SQL querying using:

   * SELECT
   * WHERE
   * ORDER BY
   * LIMIT
   * DISTINCT
   * BETWEEN
   * JOIN
8. Reading SQL results using pandas.
9. Reproducing the SQL JOIN using `pandas.merge()`.
10. Comparing SQL JOIN and pandas merge results.

## Database

The SQLite database contains:

### `categories`

* `category_id` - Primary key
* `category_name` - Unique category name

### `books`

* `book_id` - Primary key
* `title`
* `price_gbp`
* `price_inr`
* `rating`
* `in_stock`
* `category_id` - Foreign key
* `book_url`

## Run Module 1

From the Module 1 project directory:

```powershell
cd ".\data_pipeline\Module 1 data pipeline\data_pipeline"
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the pipeline:

```powershell
python pipeline.py
```

See the complete module documentation in:

`data_pipeline/Module 1 data pipeline/data_pipeline/README.md`

---

# Module 2 - Analytics & Machine Learning

## Objective

Perform exploratory analysis, preprocessing, classification, class-imbalance comparison, Random Forest tuning, and regression using the Titanic dataset.

## Data Analysis

The module includes:

* Missing-value handling
* Age and fare distribution analysis
* IQR-based outlier analysis
* Survival analysis by sex
* Survival analysis by passenger class
* Combined sex and passenger-class analysis
* Correlation analysis
* Feature standardization

## Classification

The modeling pipeline uses:

* Median imputation for numeric features
* Most-frequent imputation for categorical features
* One-hot encoding
* StandardScaler
* Stratified train/test splitting
* Logistic Regression
* Decision Tree
* Random Forest

The saved classification results are available in:

`analytics/Module 2 analytics/outputs/classification_results.csv`

## Class Imbalance

The following strategies are compared:

* Baseline
* Class-weight balancing
* SMOTE

Results are saved in:

`analytics/Module 2 analytics/outputs/imbalance_results.csv`

## Random Forest Tuning

Random Forest hyperparameters are tuned using grid search.

The saved best pipeline is:

`analytics/Module 2 analytics/models/titanic_best_pipeline.joblib`

## Regression

A multivariate Linear Regression model is used to predict fare.

Regression results are saved in:

`analytics/Module 2 analytics/outputs/regression_results.csv`

## Run Module 2

From the Module 2 directory:

```powershell
cd ".\analytics\Module 2 analytics"
```

Install the required packages according to the project's environment and run:

```powershell
python analytics_pipeline.py
```

See the complete analysis and results in:

`analytics/Module 2 analytics/README.md`

---

# Module 3 - Zepto GenAI Support Assistant

## Objective

Build a policy-document-grounded RAG service capable of answering questions about Zepto policies.

## Architecture

```text
User Query
    |
    v
FastAPI /ask
    |
    v
LangGraph classify_intent
    |
    +---------------------------+
    |                           |
    v                           v
policy_question          general_question
    |                           |
    v                           v
Retrieve top 3            direct_answer
    |
    v
Mock or real generation
    |
    v
Pydantic validation
    |
    v
JSON response
```

## Main Components

### Document ingestion

Eight policy documents are stored in the `docs/` directory.

Documents are loaded and chunked by the RAG pipeline.

### Embeddings

The system uses the local:

```text
all-MiniLM-L6-v2
```

Sentence-transformer model.

### Vector database

ChromaDB stores document chunks and their embeddings in the:

```text
zepto_policy_chunks
```

collection.

### Retrieval

The system retrieves the top three most similar document chunks using cosine distance.

### LangGraph

The workflow contains:

* `classify_intent`
* `retrieve_and_answer`
* `direct_answer`

### FastAPI

The API exposes the `/ask` endpoint.

Swagger documentation is available when the application is running at:

```text
http://127.0.0.1:8000/docs
```

## Mock LLM Mode

The graded baseline uses:

```text
MOCK_LLM=1
```

This mode does not require an external LLM API or paid service.

Policy questions use retrieved context and a deterministic response template.

General questions return a fixed response indicating that the assistant answers Zepto policy questions.

## Run Module 3 Locally

From the Module 3 directory:

```powershell
cd ".\support_assistant\Module 3 support assistant"
```

Create and activate a virtual environment if required:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Build the document index:

```powershell
python -m src.rag
```

Start the FastAPI service:

```powershell
uvicorn src.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build the Docker image:

```powershell
docker build -t zepto-genai .
```

Run the container:

```powershell
docker run --rm -p 7860:7860 zepto-genai
```

---

# Technologies Used

## Module 1

* Python
* Requests
* BeautifulSoup
* pandas
* SQLite
* SQL
* Git

## Module 2

* Python
* pandas
* NumPy
* Seaborn
* Matplotlib
* scikit-learn
* imbalanced-learn
* Joblib

## Module 3

* Python
* FastAPI
* Pydantic
* LangGraph
* ChromaDB
* Sentence Transformers
* Docker

---

# Security and Generated Files

Sensitive configuration files such as `.env` are excluded from Git.

Generated ChromaDB files are also excluded from version control.

API keys, if used for the optional real-LLM extension, should be stored as environment variables and should never be committed to the repository.

---

# Project Documentation

Each module contains its own detailed README:

* **Module 1:** `data_pipeline/Module 1 data pipeline/data_pipeline/README.md`
* **Module 2:** `analytics/Module 2 analytics/README.md`
* **Module 3:** `support_assistant/Module 3 support assistant/README.md`

---

# Git Workflow

The project uses a Git-based workflow with feature development merged into the `main` branch.

The repository contains the complete capstone project in a single GitHub repository.

---

## Author

**Gopi Chand**

Masai AIML - IIT Patna

## Repository

GitHub:

`https://github.com/yynvgopichand/zepto-data-AI-platform`
