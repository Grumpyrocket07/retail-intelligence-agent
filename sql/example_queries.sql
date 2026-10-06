-- Sanity and spot-check queries. etl/validate.py runs and prints each block.
-- They also serve as seed questions for the Text-to-SQL test set in milestone 4.

-- name: Top 5 countries by revenue
SELECT c.country_name, ROUND(SUM(oi.line_total), 2) AS revenue
FROM order_items oi
JOIN orders o ON o.invoice_no = oi.invoice_no
JOIN countries c ON c.country_id = o.country_id
GROUP BY c.country_name
ORDER BY revenue DESC
LIMIT 5;

-- name: Revenue by month (first 6 months)
SELECT DATE_FORMAT(o.order_date, '%Y-%m') AS month, ROUND(SUM(oi.line_total), 2) AS revenue
FROM order_items oi
JOIN orders o ON o.invoice_no = oi.invoice_no
GROUP BY month
ORDER BY month
LIMIT 6;

-- name: Top 5 products by units sold
SELECT p.stock_code, p.description, SUM(oi.quantity) AS units
FROM order_items oi
JOIN products p ON p.stock_code = oi.stock_code
GROUP BY p.stock_code, p.description
ORDER BY units DESC
LIMIT 5;

-- name: Top 5 customers by number of orders
SELECT customer_id, COUNT(*) AS orders
FROM orders
WHERE customer_id IS NOT NULL
GROUP BY customer_id
ORDER BY orders DESC
LIMIT 5;

-- name: Share of orders that are anonymous
SELECT ROUND(100 * SUM(customer_id IS NULL) / COUNT(*), 1) AS anonymous_pct
FROM orders;

-- name: Products with the most returned units
SELECT p.stock_code, p.description, SUM(r.quantity) AS units_returned
FROM returns r
JOIN products p ON p.stock_code = r.stock_code
GROUP BY p.stock_code, p.description
ORDER BY units_returned DESC
LIMIT 5;
