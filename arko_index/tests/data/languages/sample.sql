-- Orders schema.
CREATE TABLE customers (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);

CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES customers (id),
    total NUMERIC(12, 2) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX orders_customer_idx ON orders (customer_id);

/* Totals per customer. */
CREATE VIEW customer_totals AS
SELECT c.id, c.name, sum(o.total) AS total
FROM customers c
JOIN orders o ON o.customer_id = c.id
GROUP BY c.id, c.name;

INSERT INTO customers (name, email) VALUES ('Ada', 'ada@example.com');

UPDATE orders SET total = total * 1.1 WHERE created_at < now() - INTERVAL '1 year';

SELECT name, total
FROM customer_totals
WHERE total > 100
ORDER BY total DESC
LIMIT 10;

DELETE FROM orders WHERE total = 0;
