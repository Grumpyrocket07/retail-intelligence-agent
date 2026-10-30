# Putting the demo online (Hugging Face Space + AWS RDS)

Not yet tested end to end by the author: these steps use features of Hugging Face and AWS that were not run during development.
If anything fails, copy the error text and ask for help rather than guessing.

What runs where:

| Piece | Where | Why |
|---|---|---|
| The web app (Streamlit), agent, churn model, document search | Hugging Face Space (free CPU, 16 GB memory) | Streamlit Community Cloud has too little memory for PyTorch and two language models |
| The 1M-row MySQL database | AWS RDS (the free-tier instance already used for the ETL project) | The app needs a database that is reachable from the internet |
| The 250 MB sentiment model | Hugging Face Hub (model repository) | Too big for git; the Space downloads it on first use |
| The AI that chooses tools and writes SQL | Gemini free tier | Your API key goes into the Space's secrets, never into git |

## A. Put the database on AWS RDS

1. AWS console, region **ap-southeast-2**, RDS, Databases: open your existing database and copy its **Endpoint** (it ends in `.rds.amazonaws.com`).
2. Check the security group allows inbound **MySQL/Aurora, TCP 3306**. Hugging Face has no fixed address, so it must accept connections from anywhere (`0.0.0.0/0`).
   The ETL dashboard on Streamlit Cloud already needed the same. The protection is strong passwords and a user that can only read.
3. Create the database and two users (replace the three YOUR-... values and choose two passwords with letters and numbers only):
```
mysql -h YOUR-RDS-ENDPOINT -P 3306 -u YOUR-ADMIN-USER -p -e "CREATE DATABASE IF NOT EXISTS retail_agent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci; CREATE USER IF NOT EXISTS 'retail_loader'@'%' IDENTIFIED BY 'LoaderPassword123'; GRANT ALL PRIVILEGES ON retail_agent.* TO 'retail_loader'@'%'; CREATE USER IF NOT EXISTS 'retail_ro'@'%' IDENTIFIED BY 'ReadOnlyPassword456'; GRANT SELECT ON retail_agent.* TO 'retail_ro'@'%'; FLUSH PRIVILEGES;"
```
4. Load the data into RDS. First keep a copy of your local settings, then open `.env` and change the five database lines to the RDS values
   (`DB_HOST` = the endpoint, `DB_PORT=3306`, `DB_USER=retail_loader` and its password, `DB_NAME=retail_agent`, `DB_RO_USER=retail_ro` and its password):
```
cd C:\Users\prabh\OneDrive\Desktop\retail-intelligence-agent
.venv\Scripts\activate
copy .env .env.local
notepad .env
python -m etl.load --reset
python -m etl.validate
python -m sql_agent.ask "How many orders are in the database?"
```
   The load takes much longer than on your own computer (plan for 20 to 40 minutes). `validate` must say 20/20 checks passed and the last command must print 39516.
   When finished, restore your local settings with `copy .env.local .env` (or keep the RDS ones while testing).

## B. Put the sentiment model on the Hugging Face Hub

1. Create a free account at huggingface.co, then Settings, Access Tokens, New token, type **Write**. Copy the token.
2. Upload the model (replace YOUR-HF-USERNAME and paste the token; this takes a few minutes):
```
cd C:\Users\prabh\OneDrive\Desktop\retail-intelligence-agent
.venv\Scripts\activate
set HF_TOKEN=PASTE-YOUR-TOKEN
python -c "from huggingface_hub import HfApi; a=HfApi(); a.create_repo('YOUR-HF-USERNAME/retail-sentiment-distilbert', exist_ok=True); a.upload_folder(folder_path='models/sentiment_model', repo_id='YOUR-HF-USERNAME/retail-sentiment-distilbert')"
```

## C. Create the Space and upload the app

1. On huggingface.co: New, Space. Name `retail-intelligence-platform`, SDK **Streamlit**, hardware **CPU basic (free)**, **Public**. Create it.
2. Assemble the upload folder. It copies only code, the documents and the small churn model files, and stops if it finds anything secret:
```
cd C:\Users\prabh\OneDrive\Desktop\retail-intelligence-agent
.venv\Scripts\activate
python deploy/build_space.py
```
   It must print "Nothing secret or oversized was copied."
3. Push that folder to the Space. When git asks for a username, give your Hugging Face username; for the password, paste your token:
```
cd build\space
git init -b main
git config user.name "YOUR NAME"
git config user.email "YOUR-EMAIL"
git add .
git commit -m "Deploy Retail Intelligence Platform"
git remote add space https://huggingface.co/spaces/YOUR-HF-USERNAME/retail-intelligence-platform
git push --force space main
```
4. In the Space: Settings, **Variables and secrets**, and add each of these as a **secret**:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | your Gemini key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` (the default; every evaluation in the README used it) |
| `DB_HOST` | the RDS endpoint |
| `DB_PORT` | `3306` |
| `DB_NAME` | `retail_agent` |
| `DB_RO_USER` | `retail_ro` |
| `DB_RO_PASSWORD` | the read-only password |
| `SENTIMENT_MODEL_PATH` | `YOUR-HF-USERNAME/retail-sentiment-distilbert` |
| `APP_SESSION_LIMIT` / `APP_DAILY_LIMIT` | optional: questions per visitor per 10 minutes (default 8) and per day for everyone (default 120) |

   Do not add the loader user or its password: the app never needs it.
5. The Space builds by itself (the first build takes several minutes: look at the **Logs** tab). When it says Running, open the app.
   The first start also builds the search index and downloads a small embedding model; the first sentiment question downloads the sentiment model.

## D. Check it

1. Ask the example questions in the sidebar. Open "What happened" under an answer: it should list the tool, its time and, for database questions, the SQL.
2. Open the Monitoring page: the numbers should have changed.
3. Put the Space address in the README and on the resume as the live demo link.

## Things to know

- A free Space goes to sleep after a period without visitors and wakes in about a minute when someone opens it.
- The trace file lives inside the Space and is lost when it restarts. That is fine for a demo; the Monitoring page only shows the current run.
- The daily and per-visitor limits protect your free Gemini quota. When the quota is used up the app says so politely instead of failing.
- If the key ever appears in a log, a screenshot or git, create a new key at aistudio.google.com and replace the secret.
- Free-tier RDS is free for 12 months for new accounts. Set a billing alert in AWS so a surprise charge cannot happen.
