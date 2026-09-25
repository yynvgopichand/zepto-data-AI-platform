-- query_1_select_where
SELECT title, rating
            FROM books
            WHERE rating = 5

-- query_2_order_by
SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC

-- query_3_order_by_limit
SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10

-- query_4_distinct
SELECT DISTINCT category_id
            FROM books

-- query_5_between
SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 20 AND 40

-- query_6_join
SELECT
                b.title,
                c.category_name,
                b.rating
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.title ASC
            LIMIT 10
