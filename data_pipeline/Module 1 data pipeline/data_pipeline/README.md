
# Books Data Pipeline

## Project Overview

This project implements a data pipeline using the Books to Scrape website.

The pipeline performs the following steps:

1. Scrapes book information using requests and BeautifulSoup.
2. Extracts book title, price, rating, availability, category, and URL.
3. Cleans and transforms the scraped data using pandas.
4. Converts price from GBP to INR using the fixed rate of 105.50.
5. Creates a normalized SQLite database.
6. Stores categories and books in separate relational tables.
7. Executes SQL queries using SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, and JOIN.
8. Reads SQL results using pandas.
9. Reproduces a SQL JOIN using pandas merge.
10. Compares SQL JOIN and pandas merge results.

## Technologies Used

- Python
- requests
- BeautifulSoup
- pandas
- SQLite
- SQL
- Git

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the pipeline from the project root:

```bash
python data_pipeline/pipeline.py
```

## Data Cleaning

The following cleaning steps were performed:

- Converted price values into numeric GBP values.
- Converted rating words into numbers from 1 to 5.
- Converted availability into a Boolean in_stock field.
- Removed records with missing essential values.
- Converted GBP price to INR using the fixed conversion rate of 105.50.
- Rounded INR values to two decimal places.
- Validated that at least 60 books and 3 categories were available.

## Database Design

The database contains two tables:

### categories

- category_id: Primary key
- category_name: Unique category name

### books

- book_id: Primary key
- title: Book title
- price_gbp: Price in GBP
- price_inr: Price in INR
- rating: Rating from 1 to 5
- in_stock: Stock status
- category_id: Foreign key referencing categories
- book_url: Book detail URL

The categories and books tables are related using category_id.

## SQL Queries

The project includes SQL queries using:

- SELECT and WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- JOIN

The SQL statements are saved in queries.sql and their results are saved in the query_outputs directory.

## Pandas Analysis

The project uses pandas read_sql_query to read database results.

The SQL JOIN is reproduced using pandas merge. Both results are sorted consistently and compared for equality.

The comparison is saved in join_comparison.csv.

## Outputs

The pipeline generates:

- books_scraped.csv
- cleaned_books.csv
- books.db
- queries.sql
- query_outputs
- join_comparison.csv