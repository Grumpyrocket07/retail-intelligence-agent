-- Run once as a privileged user (e.g. root). The AI-written queries run under THIS user, which can only read.
-- Pick your own password (letters and numbers only keeps the Windows command line and .env file happy).
CREATE USER IF NOT EXISTS 'retail_ro'@'localhost' IDENTIFIED BY 'CHOOSE_A_PASSWORD';
GRANT SELECT ON retail_agent.* TO 'retail_ro'@'localhost';
FLUSH PRIVILEGES;
