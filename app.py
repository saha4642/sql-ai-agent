import os
import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="NL → SQL Agent (MySQL)",
    layout="wide"
)

MYSQL_URL = os.getenv("MYSQL_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not MYSQL_URL:
    st.error("MYSQL_URL not set. Add it in Railway variables.")
    st.stop()

# ----------------------------
# DB Engine
# ----------------------------
@st.cache_resource
def get_engine():
    return create_engine(MYSQL_URL, pool_pre_ping=True)

engine = get_engine()

# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0,
)

# ----------------------------
# Helpers
# ----------------------------
SQL_BLOCK = re.compile(r"```sql(.*?)```", re.S | re.I)

def extract_sql(text):
    m = SQL_BLOCK.search(text)
    return m.group(1).strip() if m else text.strip()

def is_safe_sql(sql):
    sql = sql.lower()
    blocked = ["insert", "update", "delete", "drop", "alter", "create", "grant"]
    return sql.startswith("select") and not any(b in sql for b in blocked)

def get_schema():
    q = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
    ORDER BY table_name, ordinal_position
    """
    df = pd.read_sql(q, engine)
    out = []
    for t, g in df.groupby("table_name"):
        cols = ", ".join(f"{r.column_name}({r.data_type})" for r in g.itertuples())
        out.append(f"{t}: {cols}")
    return "\n".join(out)

# ----------------------------
# UI
# ----------------------------
st.title("Natural Language → SQL Agent (MySQL)")

question = st.text_area(
    "Ask a question",
    placeholder="Which country's customers spent the most by invoice?"
)

if st.button("Run Query"):
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        st.success("Database connection OK")
    except Exception as e:
        st.error(f"DB connection failed: {e}")
        st.stop()

    schema = get_schema()

    prompt = f"""
You are a senior data analyst.
Write a single MySQL SELECT query.

Schema:
{schema}

Question:
{question}

Return ONLY SQL.
"""

    sql = extract_sql(llm.invoke(prompt).content)

    if not is_safe_sql(sql):
        st.error("Unsafe SQL generated")
        st.stop()

    st.code(sql, language="sql")

    df = pd.read_sql(text(sql), engine)
    st.dataframe(df, use_container_width=True)
