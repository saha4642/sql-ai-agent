from __future__ import annotations

import os
import re
from typing import Tuple

import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error as MySQLError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_openai import ChatOpenAI

# ----------------------------
# Regex / Guards
# ----------------------------
SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def build_mysql_uri(user: str, password: str, host: str, port: str, db: str) -> str:
    return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db}"


def get_llm(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)


def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out)
    if m:
        return m.group(1).strip()
    return text_out.strip()


def is_read_only_sql(sql: str) -> bool:
    """
    Strong read-only guard:
    - Must start with SELECT or WITH
    - Must not contain any write/ddl keywords
    """
    s = sql.strip()
    low = re.sub(r"\s+", " ", s.lower())

    if not (low.startswith("select") or low.startswith("with")):
        return False

    blocked = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
        "grant", "revoke", "commit", "rollback",
    ]
    return not any(re.search(rf"\b{k}\b", low) for k in blocked)


def normalize_sql(sql: str) -> str:
    """
    Basic normalization:
    - Trim
    - Ensure single trailing semicolon
    """
    s = sql.strip()
    s = s.rstrip(";").strip()
    return s + ";"


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    """
    Adds LIMIT <default_limit> only if:
    - It's a SELECT query
    - AND there is no existing LIMIT in the query
    - AND it's not a pure aggregate COUNT query
    """
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())

    # If user already included a LIMIT, don't add another
    if re.search(r"\blimit\s+\d+\b", low):
        return s

    # Add a default LIMIT for non-count SELECTs
    if low.startswith("select") and ("count(" not in low):
        s = s.rstrip(";") + f" LIMIT {default_limit};"
        return s

    return s


def fetch_schema_summary(engine: Engine, max_tables: int = 200) -> str:
    """
    Compact schema from INFORMATION_SCHEMA for prompting.
    """
    q = text("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    db = engine.url.database
    df = pd.read_sql(q, engine, params={"db": db})

    lines = []
    for tname, g in df.groupby("TABLE_NAME"):
        cols = ", ".join([f"{r.COLUMN_NAME}({r.DATA_TYPE})" for r in g.itertuples(index=False)])
        lines.append(f"- {tname}: {cols}")
        if len(lines) >= max_tables:
            lines.append("... (schema truncated)")
            break
    return "\n".join(lines)


def generate_sql_from_question(llm: ChatOpenAI, schema: str, question: str) -> str:
    """
    Ask LLM to produce SQL ONLY.
    Then normalize + apply LIMIT safely (no duplicates).
    """
    system = (
        "You are a senior data analyst writing MySQL queries.\n"
        "Rules:\n"
        "- Return ONLY the SQL query (no explanation).\n"
        "- SQL MUST be read-only (SELECT or WITH only).\n"
        "- Use the provided schema.\n"
        "- If ambiguous, make a reasonable assumption.\n"
    )

    user = (
        f"SCHEMA:\n{schema}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Return ONLY SQL."
    )

    raw = llm.invoke([{"role": "system", "content": system},
                      {"role": "user", "content": user}]).content

    sql = extract_sql(raw)
    sql = normalize_sql(sql)
    sql = ensure_limit(sql, default_limit=200)
    return sql


def run_sql_to_df(engine: Engine, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)


def create_database_mysql(host: str, port: int, user: str, password: str, db_name: str) -> None:
    conn = mysql.connector.connect(host=host, port=port, user=user, password=password)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
    conn.commit()
    cur.close()
    conn.close()


def import_sql_file_mysql(host: str, port: int, user: str, password: str, db_name: str, sql_text: str) -> Tuple[int, int]:
    """
    Import a .sql file into db_name using mysql-connector multi=True.
    Returns (ok_statements, failed_statements).

    Note: Complex scripts using DELIMITER/procedures may fail. In that case,
    use MySQL Shell/CLI SOURCE.
    """
    conn = mysql.connector.connect(
        host=host, port=port, user=user, password=password, database=db_name, autocommit=False
    )
    cur = conn.cursor()

    ok = 0
    fail = 0

    try:
        for _ in cur.execute(sql_text, multi=True):
            ok += 1
        conn.commit()
    except MySQLError:
        conn.rollback()
        fail += 1
        raise
    finally:
        cur.close()
        conn.close()

    return ok, fail


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

with st.sidebar:
    st.header("OpenAI")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    st.divider()
    st.header("MySQL Connection")
    mysql_host = st.text_input("Host", value=os.getenv("MYSQL_HOST", "127.0.0.1"))
    mysql_port = st.text_input("Port", value=os.getenv("MYSQL_PORT", "3306"))
    mysql_user = st.text_input("User", value=os.getenv("MYSQL_USERNAME", "root"))
    mysql_pass = st.text_input("Password", type="password", value=os.getenv("MYSQL_PASSWORD", ""))

    st.divider()
    st.header("Database")
    db_name = st.text_input("Database name", value=os.getenv("MYSQL_DATABASE", "Chinook"))

    c1, c2 = st.columns(2)
    with c1:
        do_create_db = st.button("Create DB")
    with c2:
        do_test_conn = st.button("Test Connection")

    st.divider()
    st.subheader("Optional: Import .sql into DB")
    sql_upload = st.file_uploader("Upload a .sql file", type=["sql"])
    do_import = st.button("Import SQL File")

# --- Sidebar actions
if do_create_db:
    try:
        create_database_mysql(mysql_host, int(mysql_port), mysql_user, mysql_pass, db_name)
        st.sidebar.success(f"Database `{db_name}` created (or already exists).")
    except Exception as e:
        st.sidebar.error(f"Create DB failed: {e}")

if do_test_conn:
    try:
        engine = create_engine(build_mysql_uri(mysql_user, mysql_pass, mysql_host, mysql_port, db_name))
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.sidebar.success("Connection OK ✅")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

if do_import:
    if not sql_upload:
        st.sidebar.warning("Upload a .sql file first.")
    else:
        try:
            sql_text = sql_upload.getvalue().decode("utf-8", errors="replace")
            ok, _ = import_sql_file_mysql(mysql_host, int(mysql_port), mysql_user, mysql_pass, db_name, sql_text)
            st.sidebar.success(f"Import finished. Statements executed: {ok}.")
        except Exception as e:
            st.sidebar.error(f"Import failed: {e}")
            st.sidebar.info("Tip: If the .sql uses DELIMITER/procedures, import using MySQL Shell SOURCE instead.")

# ----------------------------
# Main: Ask question -> SQL -> DataFrame
# ----------------------------
st.subheader("Ask a question")
question = st.text_area(
    "Example: Which country's customers spent the most by invoice?",
    height=90,
)

run_btn = st.button("Run Query")

if run_btn:
    if not api_key.strip():
        st.error("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

    # Build DB engine
    try:
        engine = create_engine(build_mysql_uri(mysql_user, mysql_pass, mysql_host, mysql_port, db_name))
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

    llm = get_llm(api_key=api_key, model=model, temperature=0.0)

    with st.spinner("Reading schema..."):
        schema = fetch_schema_summary(engine)

    with st.spinner("Generating SQL..."):
        sql = generate_sql_from_question(llm, schema, question)

    st.markdown("### Generated SQL")
    st.code(sql, language="sql")

    if not is_read_only_sql(sql):
        st.error("Blocked: generated SQL is not read-only (SELECT/WITH only).")
        st.stop()

    with st.spinner("Running query..."):
        try:
            df = run_sql_to_df(engine, sql)
        except Exception as e:
            st.error(f"SQL execution failed: {e}")
            st.stop()

    st.markdown("### Results (DataFrame)")
    st.dataframe(df, use_container_width=True)

    # Optional: Summarize results for business users
    with st.spinner("Summarizing results..."):
        preview = df.head(20).to_csv(index=False)
        summary_prompt = (
            "Summarize the result for a business user in 2-5 bullet points.\n"
            "If there are totals/rankings, mention the top items.\n\n"
            f"Question: {question}\n\n"
            f"SQL:\n{sql}\n\n"
            f"CSV Preview (first rows):\n{preview}"
        )
        summary = llm.invoke(summary_prompt).content

    st.markdown("### Summary")
    st.write(summary)
