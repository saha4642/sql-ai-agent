# app.py
from __future__ import annotations

import os
import re
from typing import Tuple
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError

from langchain_openai import ChatOpenAI

# ============================================================
# 0) Railway-safe MySQL connection utilities
#    - Prefer MYSQL_PUBLIC_URL (non-root)
#    - Fall back to MYSQL_USER/PASSWORD/HOST/PORT/DB (non-root)
#    - Never allow root from the app
#    - Encode creds + force mysql+mysqlconnector dialect
# ============================================================

def env_any(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v)
    return ""


def normalize_mysql_sqlalchemy_url(raw: str) -> str:
    """
    Convert Railway/standard MySQL URLs into a valid SQLAlchemy URL:
      - mysql:// -> mysql+mysqlconnector://
      - mysql+mysqldb:// -> mysql+mysqlconnector://
      - strip quotes
      - URL-encode username/password
      - validate via SQLAlchemy make_url
    """
    if not raw:
        raise ValueError("Database URL is empty/None. Check Railway environment variables.")

    s = str(raw).strip().strip('"').strip("'")

    if s.startswith("mysql://"):
        s = s.replace("mysql://", "mysql+mysqlconnector://", 1)
    elif s.startswith("mysql+mysqldb://"):
        s = s.replace("mysql+mysqldb://", "mysql+mysqlconnector://", 1)

    u = make_url(s)

    # Encode credentials to handle special chars
    username = quote_plus(u.username) if u.username else None
    password = quote_plus(u.password) if u.password else None
    u = u.set(username=username, password=password)

    return str(u)


def build_url_from_discrete_vars() -> str:
    """
    Supports both Railway naming styles:
      - MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE / MYSQL_DB / MYSQL_DEFAULT_DB
      - MYSQLUSER, MYSQLPASSWORD, MYSQLHOST, MYSQLPORT, MYSQLDATABASE
    """
    user = env_any("MYSQL_USER", "MYSQLUSER")
    pw = env_any("MYSQL_PASSWORD", "MYSQLPASSWORD")
    host = env_any("MYSQL_HOST", "MYSQLHOST")
    port = env_any("MYSQL_PORT", "MYSQLPORT")
    db = env_any(
        "MYSQL_DATABASE", "MYSQLDATABASE",
        "MYSQL_DB", "MYSQL_DEFAULT_DB"
    )

    if all([user, pw, host, port, db]):
        return f"mysql://{user}:{pw}@{host}:{port}/{db}"
    return ""


def pick_best_mysql_url() -> str:
    """
    Railway best practice for apps:
      1) MYSQL_PUBLIC_URL (preferred, non-root)
      2) MYSQL_PUBLIC_URL (if user uses variant naming)
      3) Discrete vars MYSQL_USER/PASSWORD/HOST/PORT/DB
      4) MYSQL_URL (often root) -> will be blocked if root
    """
    return (
        env_any("MYSQL_PUBLIC_URL", "MYSQL_PUBLIC_URL")
        or build_url_from_discrete_vars()
        or env_any("MYSQL_URL")
        or ""
    )


def assert_not_root(sqlalchemy_url: str, label: str = "DB URL") -> None:
    u = make_url(sqlalchemy_url)
    if (u.username or "").lower() == "root":
        raise ValueError(
            f"{label} is using user 'root'. Railway blocks root for app connections.\n"
            "Use MYSQL_PUBLIC_URL (recommended) or MYSQL_USER/MYSQL_PASSWORD variables."
        )


@st.cache_resource(show_spinner=False)
def get_engine(sqlalchemy_url: str) -> Engine:
    # Cache is keyed by sqlalchemy_url string; changing it creates a new engine.
    return create_engine(sqlalchemy_url, pool_pre_ping=True, pool_recycle=1800)


def test_engine(engine: Engine) -> Tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ============================================================
# 1) SQL safety + LLM helpers
# ============================================================

SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out or "")
    return m.group(1).strip() if m else (text_out or "").strip()

def normalize_sql(sql: str) -> str:
    s = (sql or "").strip()
    s = s.rstrip(";").strip()
    return s + ";"

def is_read_only_sql(sql: str) -> bool:
    low = re.sub(r"\s+", " ", (sql or "").strip().lower())
    if not (low.startswith("select") or low.startswith("with")):
        return False
    blocked = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
        "grant", "revoke", "commit", "rollback", "call", "load data", "outfile"
    ]
    return not any(re.search(rf"\b{re.escape(k)}\b", low) for k in blocked)

def ensure_limit(sql: str, default_limit: int = 200) -> str:
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())
    if re.search(r"\blimit\s+\d+\b", low):
        return s
    if low.startswith("select") and ("count(" not in low):
        return s.rstrip(";") + f" LIMIT {default_limit};"
    return s

def get_llm(api_key: str, model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)

def fetch_schema_summary(engine: Engine, db: str, max_tables: int = 200) -> str:
    q = text("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    df = pd.read_sql(q, engine, params={"db": db})

    lines = []
    for tname, g in df.groupby("TABLE_NAME"):
        cols = ", ".join(f"{r.COLUMN_NAME}({r.DATA_TYPE})" for r in g.itertuples(index=False))
        lines.append(f"- {tname}: {cols}")
        if len(lines) >= max_tables:
            lines.append("... (schema truncated)")
            break
    return "\n".join(lines)

def generate_sql_from_question(llm: ChatOpenAI, schema: str, question: str) -> str:
    system = (
        "You are a senior data analyst writing MySQL queries.\n"
        "Rules:\n"
        "- Return ONLY the SQL query (no explanation).\n"
        "- SQL MUST be read-only (SELECT or WITH only).\n"
        "- Use only tables/columns from the provided schema.\n"
        "- If ambiguous, make a reasonable assumption.\n"
    )
    user = f"SCHEMA:\n{schema}\n\nQUESTION:\n{question}\n\nReturn ONLY SQL."

    raw = llm.invoke(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}]
    ).content

    sql = extract_sql(raw)
    sql = normalize_sql(sql)
    sql = ensure_limit(sql, default_limit=200)
    return sql

def run_sql_to_df(engine: Engine, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)


# ============================================================
# 2) Streamlit UI
# ============================================================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

ENV_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

with st.sidebar:
    st.header("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_model = st.text_input("OPENAI_MODEL", value=ENV_OPENAI_MODEL)

    st.divider()
    st.header("MySQL (Railway)")
    st.caption(
        "This app automatically uses Railway's MYSQL_PUBLIC_URL (non-root). "
        "If missing, it uses MYSQL_USER/MYSQL_PASSWORD/etc. Root is blocked."
    )

    col1, col2 = st.columns(2)
    if col1.button("Clear DB Cache"):
        st.cache_resource.clear()
        st.rerun()
    btn_test = col2.button("Test Connection")

# Pick URL from env only (no manual input -> prevents pasting root)
raw_url = pick_best_mysql_url()
if not raw_url:
    st.error(
        "No MySQL connection info found.\n\n"
        "In Railway, add variable references from your MySQL service into this app service:\n"
        "- MYSQL_PUBLIC_URL (recommended)\n"
        "or\n"
        "- MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE\n"
    )
    st.stop()

try:
    sqlalchemy_url = normalize_mysql_sqlalchemy_url(raw_url)
    assert_not_root(sqlalchemy_url, label="Admin/app URL")
except Exception as e:
    st.error(str(e))
    st.stop()

parsed = make_url(sqlalchemy_url)
db_name = parsed.database or env_any("MYSQLDATABASE", "MYSQL_DATABASE", "MYSQL_DB", "MYSQL_DEFAULT_DB") or "railway"

engine = get_engine(sqlalchemy_url)

with st.sidebar.expander("DB Debug (safe)", expanded=False):
    st.write("Using user:", parsed.username)
    st.write("Host:", parsed.host)
    st.write("Port:", parsed.port)
    st.write("Database:", db_name)
    st.write("MYSQL_PUBLIC_URL set:", bool(env_any("MYSQL_PUBLIC_URL", "MYSQL_PUBLIC_URL")))
    st.write("MYSQL_URL set:", bool(env_any("MYSQL_URL")))
    st.write("Discrete vars set:", bool(build_url_from_discrete_vars()))

if btn_test:
    ok, msg = test_engine(engine)
    if ok:
        st.sidebar.success("Connection OK ✅")
    else:
        st.sidebar.error(f"Connection failed: {msg}")

with st.expander("Connection status", expanded=True):
    ok, msg = test_engine(engine)
    st.write(f"**Engine:** {'✅ OK' if ok else '❌ FAIL'}")
    if not ok:
        st.code(msg)

# ============================================================
# 3) Main: Ask question -> SQL -> DataFrame
# ============================================================

st.subheader("Ask a question")
question = st.text_area(
    "Example: Which country's customers spent the most by invoice?",
    height=90,
)

run_btn = st.button("Run Query")

if run_btn:
    if not openai_key.strip():
        st.error("Missing OPENAI_API_KEY. Set it in Railway variables or enter it in the sidebar.")
        st.stop()

    ok, msg = test_engine(engine)
    if not ok:
        st.error(f"Database connection failed: {msg}")
        st.stop()

    llm = get_llm(api_key=openai_key, model=openai_model, temperature=0.0)

    with st.spinner("Reading schema..."):
        try:
            schema = fetch_schema_summary(engine, db=db_name)
        except SQLAlchemyError as e:
            st.error(f"Failed to read schema for `{db_name}`: {e}")
            st.stop()

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

    with st.spinner("Summarizing results..."):
        preview = df.head(20).to_csv(index=False)
        summary_prompt = (
            "Summarize the result for a business user in 2-5 bullet points.\n"
            "If there are totals/rankings, mention the top items.\n\n"
            f"Question: {question}\n\n"
            f"SQL:\n{sql}\n\n"
            f"CSV Preview (first rows):\n{preview}"
        )
        try:
            summary = llm.invoke(summary_prompt).content
        except Exception as e:
            summary = f"(Summary failed: {e})"

    st.markdown("### Summary")
    st.write(summary)
