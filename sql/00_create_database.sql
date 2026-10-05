-- Run once as a privileged user (e.g. root).
-- utf8mb4 + utf8mb4_unicode_ci is set explicitly everywhere so foreign keys never hit a collation mismatch.
CREATE DATABASE IF NOT EXISTS retail_agent
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Give your existing ETL user access to the new database.
-- MySQL 8 will not create the user for you: if this errors, the user does not exist yet or its host
-- differs ('localhost' vs '%'). Check with: SELECT user, host FROM mysql.user;
GRANT ALL PRIVILEGES ON retail_agent.* TO 'etl_user'@'%';
FLUSH PRIVILEGES;
