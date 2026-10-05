-- Retail Intelligence Agent - milestone 1 schema (MySQL 8 / MariaDB compatible)
-- Six tables built from the cleaned UCI Online Retail II dataset.

CREATE TABLE IF NOT EXISTS countries (
  country_id   SMALLINT UNSIGNED NOT NULL,
  country_name VARCHAR(64) NOT NULL,
  PRIMARY KEY (country_id),
  UNIQUE KEY uq_countries_name (country_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS customers (
  customer_id INT UNSIGNED NOT NULL,
  country_id  SMALLINT UNSIGNED NOT NULL,
  PRIMARY KEY (customer_id),
  KEY idx_customers_country (country_id),
  CONSTRAINT fk_customers_country FOREIGN KEY (country_id) REFERENCES countries (country_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS products (
  stock_code  VARCHAR(20)  NOT NULL,
  description VARCHAR(255) NOT NULL,
  PRIMARY KEY (stock_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- One row per invoice. customer_id is NULL for anonymous (guest) invoices.
CREATE TABLE IF NOT EXISTS orders (
  invoice_no  VARCHAR(12)       NOT NULL,
  customer_id INT UNSIGNED      NULL,
  country_id  SMALLINT UNSIGNED NOT NULL,
  order_date  DATETIME          NOT NULL,
  PRIMARY KEY (invoice_no),
  KEY idx_orders_customer (customer_id),
  KEY idx_orders_date (order_date),
  KEY idx_orders_country (country_id),
  CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
  CONSTRAINT fk_orders_country  FOREIGN KEY (country_id)  REFERENCES countries (country_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- One row per invoice line. unit_price is stored here because prices change over time.
-- is_outlier flags extreme quantity/price values (kept, not dropped).
CREATE TABLE IF NOT EXISTS order_items (
  item_id    BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  invoice_no VARCHAR(12)     NOT NULL,
  stock_code VARCHAR(20)     NOT NULL,
  quantity   INT UNSIGNED    NOT NULL,
  unit_price DECIMAL(10,3)   NOT NULL,
  line_total DECIMAL(14,3)   NOT NULL,
  is_outlier TINYINT(1)      NOT NULL DEFAULT 0,
  PRIMARY KEY (item_id),
  KEY idx_items_invoice (invoice_no),
  KEY idx_items_product (stock_code),
  CONSTRAINT fk_items_order   FOREIGN KEY (invoice_no) REFERENCES orders (invoice_no),
  CONSTRAINT fk_items_product FOREIGN KEY (stock_code) REFERENCES products (stock_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Cancellations (invoice numbers starting with C). quantity is stored as a positive number.
-- The original sale is not identifiable in the source data, so there is no foreign key to orders.
CREATE TABLE IF NOT EXISTS returns (
  return_id   BIGINT UNSIGNED   NOT NULL AUTO_INCREMENT,
  invoice_no  VARCHAR(12)       NOT NULL,
  customer_id INT UNSIGNED      NULL,
  country_id  SMALLINT UNSIGNED NOT NULL,
  stock_code  VARCHAR(20)       NOT NULL,
  quantity    INT UNSIGNED      NOT NULL,
  unit_price  DECIMAL(10,3)     NOT NULL,
  line_total  DECIMAL(14,3)     NOT NULL,
  return_date DATETIME          NOT NULL,
  is_outlier  TINYINT(1)        NOT NULL DEFAULT 0,
  PRIMARY KEY (return_id),
  KEY idx_returns_invoice (invoice_no),
  KEY idx_returns_customer (customer_id),
  KEY idx_returns_product (stock_code),
  KEY idx_returns_date (return_date),
  CONSTRAINT fk_returns_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
  CONSTRAINT fk_returns_country  FOREIGN KEY (country_id)  REFERENCES countries (country_id),
  CONSTRAINT fk_returns_product  FOREIGN KEY (stock_code)  REFERENCES products (stock_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
