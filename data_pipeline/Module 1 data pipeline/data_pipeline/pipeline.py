
from pathlib import Path
import re
import sqlite3
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "books.db"
RAW_CSV = BASE_DIR / "books_scraped.csv"
CLEAN_CSV = BASE_DIR / "cleaned_books.csv"
SQL_FILE = BASE_DIR / "queries.sql"
JOIN_COMPARISON_FILE = BASE_DIR / "join_comparison.csv"
OUTPUT_DIR = BASE_DIR / "query_outputs"

BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR_RATE = 105.50

OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_soup(url):
    """Download a page and return its BeautifulSoup object."""
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_rating(rating_text):
    """Convert One, Two, Three, Four, Five into numbers."""
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    return rating_map.get(rating_text)


def parse_price(price_text):
    """Convert a price string into a float."""
    if pd.isna(price_text):
        return None

    try:
        cleaned = str(price_text)
        cleaned = cleaned.replace("Â£", "")
        cleaned = cleaned.replace("£", "")
        cleaned = cleaned.strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def parse_stock(stock_text):
    """Convert stock text into True or False."""
    if pd.isna(stock_text):
        return None

    text = str(stock_text).strip().lower()

    # Check out of stock first because the phrase
    # 'out of stock' contains the words 'in stock'.
    if "out of stock" in text:
        return False

    if text.startswith("in stock"):
        return True

    return None


# ============================================================
# 3. SCRAPE BOOK DATA
# ============================================================

def scrape_books():
    print("\nStarting web scraping...")

    all_books = []

    # Scrape the first five pages.
    for page_number in range(1, 6):

        if page_number == 1:
            page_url = BASE_URL + "index.html"
        else:
            page_url = (
                BASE_URL
                + f"catalogue/page-{page_number}.html"
            )

        print(f"Scraping page {page_number}: {page_url}")

        try:
            soup = get_soup(page_url)
        except requests.RequestException as error:
            print(f"Could not scrape page {page_number}: {error}")
            continue

        book_articles = soup.select("article.product_pod")

        for article in book_articles:

            title_element = article.select_one("h3 a")
            price_element = article.select_one(".price_color")
            availability_element = article.select_one(
                ".availability"
            )
            rating_element = article.select_one("p.star-rating")

            if not title_element:
                continue

            title = title_element.get("title", "").strip()

            price = (
                price_element.get_text(strip=True)
                if price_element
                else None
            )

            availability = (
                availability_element.get_text(" ", strip=True)
                if availability_element
                else None
            )

            rating_classes = (
                rating_element.get("class", [])
                if rating_element
                else []
            )

            rating_text = None

            for class_name in rating_classes:
                if class_name in [
                    "One",
                    "Two",
                    "Three",
                    "Four",
                    "Five"
                ]:
                    rating_text = class_name
                    break

            book_relative_url = title_element.get("href", "")
            book_url = urljoin(page_url, book_relative_url)

            # Get category from the individual book page.
            category = None

            try:
                detail_soup = get_soup(book_url)

                breadcrumbs = detail_soup.select(
                    "ul.breadcrumb li"
                )

                breadcrumb_texts = [
                    item.get_text(strip=True)
                    for item in breadcrumbs
                ]

                # Typical structure:
                # Home > Books > Category > Book title
                if len(breadcrumb_texts) >= 2:
                    category = breadcrumb_texts[-2]

            except requests.RequestException as error:
                print(
                    f"Could not open detail page for {title}: {error}"
                )

            all_books.append({
                "title": title,
                "price": price,
                "rating_text": rating_text,
                "availability": availability,
                "category": category,
                "book_url": book_url
            })

            # Small delay to avoid sending requests too quickly.
            time.sleep(0.05)

    raw_df = pd.DataFrame(all_books)

    if raw_df.empty:
        raise ValueError(
            "No books were scraped. Check your internet connection."
        )

    raw_df.to_csv(RAW_CSV, index=False)

    print(f"\nScraped {len(raw_df)} books.")
    print(f"Raw data saved to: {RAW_CSV}")

    return raw_df


# ============================================================
# 4. CLEAN AND TRANSFORM DATA
# ============================================================

def clean_data(raw_df):
    print("\nCleaning data...")

    df = raw_df.copy()

    # Convert price to numeric GBP.
    df["price_gbp"] = df["price"].apply(parse_price)

    # Convert rating words into numbers.
    df["rating"] = df["rating_text"].apply(parse_rating)

    # Convert availability text into boolean values.
    df["in_stock"] = df["availability"].apply(parse_stock)

    # Remove records missing essential fields.
    df = df.dropna(
        subset=[
            "title",
            "category",
            "price_gbp",
            "rating",
            "in_stock"
        ]
    )

    # Convert numeric fields to correct data types.
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # Calculate INR price using the required fixed rate.
    df["price_inr"] = (
        df["price_gbp"] * FIXED_GBP_TO_INR_RATE
    ).round(2)

    # Keep only the columns required for the database.
    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
            "book_url"
        ]
    ].reset_index(drop=True)

    # Validate assignment requirements.
    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} valid books found. "
            "At least 60 books are required."
        )

    if df["category"].nunique() < 3:
        raise ValueError(
            "Fewer than 3 categories were found."
        )

    if not df["rating"].between(1, 5).all():
        raise ValueError(
            "Rating values must be between 1 and 5."
        )

    df.to_csv(CLEAN_CSV, index=False)

    print(f"Cleaned records: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Cleaned data saved to: {CLEAN_CSV}")

    return df


# ============================================================
# 5. CREATE DATABASE AND LOAD DATA
# ============================================================

def create_database(df):
    print("\nCreating SQLite database...")

    # Delete the previous database to avoid duplicate records
    # when the script is run more than once.
    if DB_FILE.exists():
        DB_FILE.unlink()

    connection = sqlite3.connect(DB_FILE)

    try:
        connection.execute("PRAGMA foreign_keys = ON")

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            )
        """)

        cursor.execute("""
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL
                    CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL
                    CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                book_url TEXT,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            )
        """)

        # Insert unique categories.
        categories = sorted(
            df["category"].dropna().unique()
        )

        cursor.executemany(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            [(category,) for category in categories]
        )

        # Create category-to-ID mapping.
        category_rows = cursor.execute(
            """
            SELECT category_id, category_name
            FROM categories
            """
        ).fetchall()

        category_map = {
            category_name: category_id
            for category_id, category_name in category_rows
        }

        # Insert books.
        book_records = []

        for _, row in df.iterrows():
            book_records.append((
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_map[row["category"]],
                row["book_url"]
            ))

        cursor.executemany(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id,
                book_url
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            book_records
        )

        connection.commit()

        book_count = cursor.execute(
            "SELECT COUNT(*) FROM books"
        ).fetchone()[0]

        category_count = cursor.execute(
            "SELECT COUNT(*) FROM categories"
        ).fetchone()[0]

        print(f"Books inserted: {book_count}")
        print(f"Categories inserted: {category_count}")
        print(f"Database saved to: {DB_FILE}")

    finally:
        connection.close()


# ============================================================
# 6. RUN REQUIRED SQL QUERIES
# ============================================================

def run_sql_queries():
    print("\nRunning SQL queries...")

    connection = sqlite3.connect(DB_FILE)

    queries = {
        "query_1_select_where": """
            SELECT title, rating
            FROM books
            WHERE rating = 5
        """,

        "query_2_order_by": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
        """,

        "query_3_order_by_limit": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10
        """,

        "query_4_distinct": """
            SELECT DISTINCT category_id
            FROM books
        """,

        "query_5_between": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 20 AND 40
        """,

        "query_6_join": """
            SELECT
                b.title,
                c.category_name,
                b.rating
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.title ASC
            LIMIT 10
        """
    }

    sql_text = []

    try:
        for query_name, query in queries.items():

            result_df = pd.read_sql_query(
                query,
                connection
            )

            output_file = OUTPUT_DIR / f"{query_name}.csv"
            result_df.to_csv(output_file, index=False)

            sql_text.append(
                f"-- {query_name}\n{query.strip()}\n"
            )

            print(f"\n{query_name}")
            print(result_df.head())

        SQL_FILE.write_text(
            "\n".join(sql_text),
            encoding="utf-8"
        )

        print(f"\nSQL statements saved to: {SQL_FILE}")

    finally:
        connection.close()


# ============================================================
# 7. COMPARE SQL JOIN WITH PANDAS MERGE
# ============================================================

def compare_sql_and_pandas_join():
    print("\nComparing SQL JOIN and pandas merge...")

    connection = sqlite3.connect(DB_FILE)

    try:
        sql_join_query = """
            SELECT
                b.title,
                c.category_name,
                b.rating
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.title ASC
            LIMIT 10
        """

        sql_result = pd.read_sql_query(
            sql_join_query,
            connection
        )

        books_df = pd.read_sql_query(
            """
            SELECT
                title,
                category_id,
                rating
            FROM books
            """,
            connection
        )

        categories_df = pd.read_sql_query(
            """
            SELECT
                category_id,
                category_name
            FROM categories
            """,
            connection
        )

    finally:
        connection.close()

    pandas_result = books_df.merge(
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_result = pandas_result[
        [
            "title",
            "category_name",
            "rating"
        ]
    ]

    pandas_result = pandas_result.sort_values(
        by=["rating", "title"],
        ascending=[False, True]
    ).head(10).reset_index(drop=True)

    sql_result = sql_result.reset_index(drop=True)

    # Make sure both results have the same column order.
    pandas_result = pandas_result[
        sql_result.columns
    ]

    are_equal = sql_result.equals(pandas_result)

    side_by_side = pd.concat(
        [
            sql_result.add_prefix("sql_"),
            pandas_result.add_prefix("pandas_")
        ],
        axis=1
    )

    side_by_side.to_csv(
        JOIN_COMPARISON_FILE,
        index=False
    )

    print("\nSQL JOIN result:")
    print(sql_result)

    print("\nPandas merge result:")
    print(pandas_result)

    print(f"\nAre the results equal? {are_equal}")
    print(
        f"Side-by-side comparison saved to: "
        f"{JOIN_COMPARISON_FILE}"
    )

    if not are_equal:
        raise ValueError(
            "SQL JOIN and pandas merge results do not match."
        )


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():
    print("=" * 60)
    print("BOOKS DATA PIPELINE")
    print("=" * 60)

    raw_df = scrape_books()
    cleaned_df = clean_data(raw_df)
    create_database(cleaned_df)
    run_sql_queries()
    compare_sql_and_pandas_join()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()