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


# ============================================================
# Env helpers
# ============================================================

def env_any(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v)
    return ""


# ============================================================
# Build MySQL URL (APP_* ALWAYS wins)
# ============================================================

def build_raw_mysql_url() -> str:
    app_user = env_any("APP_MYSQL_USER").strip()
    app_pw = env_any("APP_MYSQL_PASSWORD").strip()

    rail_user = env_any("MYSQL_USER", "MYSQLUSER").strip()
    rail_pw = env_any("MYSQL_PASSWORD", "MYSQLPASSWORD").strip()

    if app_user and app_pw:
        user, pw = app_user, app_pw
    else:
        user, pw = rail_user, rail_pw

    # Railway blocks root
    if user.lower() == "root":
        return ""

    host = env_any("MYSQL_HOST", "MYSQLHOST").strip()
    port = env_any("MYSQL_PORT", "MYSQLPORT").strip()
    db = (
        env_any("MYSQL_DATABASE")
        or env_any("MYSQL_DB")
        or env_any("MYSQL_DEFAULT_DB")
        or ""
    ).strip()

    if all([user, pw, host, port, db]):
        return f"mysql://{user}:{pw}@{host}:{port}/{db}"

    return ""


def normalize_sqlalchemy_url(raw: str) -> str:
    if not raw:
        raise RuntimeError("Database URL is empty")

    s = raw.strip().strip('"').strip("'")

    # Force PyMySQL
    if s.startswith("mysql://"):
        s = s.replace("mysql://", "mysql+pymysql://", 1)
    else:
        s = re.sub(r"^mysql\+[^:]+://", "mysql+pymysql://", s)

    u = make_url(s)
    u = u.set(
        username=quote_plus(u.username) if u.username else None,
        password=quote_plus(u.password) if u.password else None,
    )
    return str(u)


# ============================================================
# SQLAlchemy engine
# ============================================================

@st.cache_resource(show_spinner=False)
def get_engine(url: str) -> Engine:
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


def test_engine(engine: Engine) -> Tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ============================================================
# SQL helpers
# ============================================================

SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_sql(txt: str) -> str:
    m = SQL_CODEBLOCK_RE.search(txt or "")
    return m.group(1).strip() if m else (txt or "").strip()


def normalize_sql(sql: str) -> str:
    return sql.rstrip(";").strip() + ";"


def is_read_only_sql(sql: str) -> bool:
    low = re.sub(r"\s+", " ", sql.lower())
    if not (low.startswith("select") or low.startswith("with")):
        return False
    banned = [
        "insert", "update", "delete", "drop", "alter", "truncate",
        "create", "replace", "grant", "revoke", "commit", "rollback",
        "call", "load data", "outfile",
    ]
    return not any(re.search(rf"\b{b}\b", low) for b in banned)


def ensure_limit(sql: str, limit: int = 200) -> str:
    low = sql.lower()
    if "limit" in low or "count(" in low:
        return sql
    return sql.rstrip(";") + f" LIMIT {limit};"


# ============================================================
# LLM helpers
# ============================================================

def get_llm(api_key: str, model: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.0)


def fetch_schema(engine: Engine, db: str) -> str:
    q = text("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    df = pd.read_sql(q, engine, params={"db": db})
    out = []
    for t, g in df.groupby("TABLE_NAME"):
        cols = ", ".join(f"{r.COLUMN_NAME}({r.DATA_TYPE})" for r in g.itertuples(index=False))
        out.append(f"- {t}: {cols}")
    return "\n".join(out)


def generate_sql(llm, schema: str, question: str) -> str:
    prompt = f"""
You are a senior data analyst writing MySQL queries.

Rules:
- Return ONLY SQL
- Read-only (SELECT / WITH)
- Use only the schema below

SCHEMA:
{schema}

QUESTION:
{question}
"""
    raw = llm.invoke(prompt).content
    sql = extract_sql(raw)
    sql = normalize_sql(sql)
    sql = ensure_limit(sql)
    return sql


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

with st.sidebar:
    st.header("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=env_any("OPENAI_API_KEY"))
    openai_model = st.text_input("OPENAI_MODEL", value=env_any("OPENAI_MODEL") or "gpt-4o-mini")

    st.divider()
    st.header("MySQL (Railway)")
    clear_cache = st.button("Clear DB Cache")
    test_btn = st.button("Test Connection")

    st.divider()
    st.header("DB Debug (safe)")
    st.write("APP_MYSQL_USER set:", bool(env_any("APP_MYSQL_USER")))
    st.write("APP_MYSQL_PASSWORD set:", bool(env_any("APP_MYSQL_PASSWORD")))
    st.write("MYSQL_HOST:", env_any("MYSQL_HOST"))
    st.write("MYSQL_PORT:", env_any("MYSQL_PORT"))
    st.write("MYSQL_DATABASE:", env_any("MYSQL_DATABASE"))

if clear_cache:
    get_engine.clear()
    st.success("DB cache cleared")

raw_url = build_raw_mysql_url()
if not raw_url:
    st.error("Invalid DB credentials or root user detected.")
    st.stop()

try:
    sqlalchemy_url = normalize_sqlalchemy_url(raw_url)
    parsed = make_url(sqlalchemy_url)
    db_name = parsed.database
except Exception as e:
    st.error(f"DB URL error: {e}")
    st.stop()

engine = get_engine(sqlalchemy_url)

# 🔴 CRITICAL DEBUG: show exactly which DB the app hits
with st.expander("Connection status", expanded=True):
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT @@hostname, @@port, DATABASE(), CURRENT_USER()")
            ).fetchone()
        st.write("APP CONNECTS TO:", row)
    except Exception as e:
        st.error(e)

    ok, msg = test_engine(engine)
    st.write(f"Engine: {'✅ OK' if ok else '❌ FAIL'}")
    if not ok:
        st.code(msg)


# ============================================================
# Main query flow
# ============================================================

st.subheader("Ask a question")
question = st.text_area(
    "Example: Which country's customers spent the most by invoice?",
    height=90,
)

if st.button("Run Query"):
    if not openai_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()

    ok, msg = test_engine(engine)
    if not ok:
        st.error(msg)
        st.stop()

    llm = get_llm(openai_key, openai_model)

    with st.spinner("Reading schema..."):
        schema = fetch_schema(engine, db_name)

    with st.spinner("Generating SQL..."):
        sql = generate_sql(llm, schema, question)

    st.markdown("### Generated SQL")
    st.code(sql, language="sql")

    if not is_read_only_sql(sql):
        st.error("Blocked non-read-only SQL")
        st.stop()

    with st.spinner("Running query..."):
        df = pd.read_sql(text(sql), engine)

    st.markdown("### Results")
    st.dataframe(df, use_container_width=True)
