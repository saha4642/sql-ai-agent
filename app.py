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
# 0) Connection helpers (Railway MySQL safe)
# ============================================================

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

    s = raw.strip().strip('"').strip("'")

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
    Railway exposes MYSQLUSER/MYSQLPASSWORD/MYSQLHOST/MYSQLPORT/MYSQLDATABASE.
    Use them as a fallback when MYSQL_PUBLIC_URL isn't used.
    """
    user = os.getenv("MYSQLUSER", "")
    pw = os.getenv("MYSQLPASSWORD", "")
    host = os.getenv("MYSQLHOST", "")
    port = os.getenv("MYSQLPORT", "")
    db = os.getenv("MYSQLDATABASE", "") or os.getenv("MYSQL_DATABASE", "")

    if all([user, pw, host, port, db]):
        # NOTE: we'll normalize+encode later
        return f"mysql://{user}:{pw}@{host}:{port}/{db}"
    return ""


def pick_best_admin_url() -> str:
    """
    Railway best practice:
      1) MYSQL_PUBLIC_URL (non-root, app-safe)
      2) discrete vars (MYSQLUSER/MYSQLPASSWORD/...)
      3) MYSQL_URL (often root; we will block root explicitly)
    """
    return (
        os.getenv("MYSQL_PUBLIC_URL", "")
        or build_url_from_discrete_vars()
        or os.getenv("MYSQL_URL", "")
        or ""
    )


def assert_not_root(url: str, label: str = "DB URL") -> None:
    """
    Railway commonly restricts root from remote/app connections.
    If user is root, fail early with a helpful message.
    """
    u = make_url(url)
    if (u.username or "").lower() == "root":
        raise ValueError(
            f"{label} is using user 'root'. Railway blocks root for app connections.\n"
            f"Use MYSQL_PUBLIC_URL (recommended) or MYSQLUSER/MYSQLPASSWORD variables."
        )


@st.cache_resource(show_spinner=False)
def get_engine(sqlalchemy_url: str) -> Engine:
    """
    Cache is keyed by the URL string, so changing URL makes a new engine.
    """
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
DEFAULT_ADMIN_RAW = pick_best_admin_url()

with st.sidebar:
    st.header("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_model = st.text_input("OPENAI_MODEL", value=ENV_OPENAI_MODEL)

    st.divider()
    st.header("MySQL Connection (Railway)")
    st.caption(
        "✅ Use MYSQL_PUBLIC_URL (recommended). "
        "🚫 Do NOT use MYSQL_URL if it connects as root."
    )

    admin_raw = st.text_input("MYSQL_PUBLIC_URL (or app-safe URL)", value=DEFAULT_ADMIN_RAW)

    colA, colB = st.columns(2)
    btn_clear_cache = colA.button("Clear DB Cache")
    btn_test = colB.button("Test Connections")

    if btn_clear_cache:
        st.cache_resource.clear()
        st.rerun()

# --- Validate + normalize admin URL
if not (admin_raw or "").strip():
    st.error("Missing DB URL. In Railway, copy MYSQL_PUBLIC_URL and paste it here.")
    st.stop()

try:
    admin_url = normalize_mysql_sqlalchemy_url(admin_raw)
    assert_not_root(admin_url, label="Admin/app URL")
except Exception as e:
    st.error(str(e))
    st.stop()

admin_parsed = make_url(admin_url)
db_name = admin_parsed.database or os.getenv("MYSQLDATABASE") or os.getenv("MYSQL_DATABASE") or "railway"

# For this app, admin/query engines can be the same (read-only safety is enforced in SQL)
engine_admin = get_engine(admin_url)
engine_query = get_engine(admin_url)

# Safe debug info (no passwords)
with st.sidebar.expander("DB Debug (safe)", expanded=False):
    st.write("Using user:", admin_parsed.username)
    st.write("Host:", admin_parsed.host)
    st.write("Port:", admin_parsed.port)
    st.write("Database:", db_name)
    st.write("MYSQL_PUBLIC_URL set:", bool(os.getenv("MYSQL_PUBLIC_URL")))
    st.write("MYSQL_URL set:", bool(os.getenv("MYSQL_URL")))
    st.write("MYSQLUSER set:", bool(os.getenv("MYSQLUSER")))

if btn_test:
    okA, msgA = test_engine(engine_admin)
    okQ, msgQ = test_engine(engine_query)
    if okA:
        st.sidebar.success("Admin/app connection OK ✅")
    else:
        st.sidebar.error(f"Admin/app connection failed: {msgA}")
    if okQ:
        st.sidebar.success("Query connection OK ✅")
    else:
        st.sidebar.error(f"Query connection failed: {msgQ}")

with st.expander("Connection status", expanded=True):
    okA, msgA = test_engine(engine_admin)
    st.write(f"**Admin/app engine:** {'✅ OK' if okA else '❌ FAIL'}")
    if not okA:
        st.code(msgA)

    okQ, msgQ = test_engine(engine_query)
    st.write(f"**Query engine:** {'✅ OK' if okQ else '❌ FAIL'}")
    if not okQ:
        st.code(msgQ)

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

    okQ, msgQ = test_engine(engine_query)
    if not okQ:
        st.error(f"Database connection failed: {msgQ}")
        st.stop()

    llm = get_llm(api_key=openai_key, model=openai_model, temperature=0.0)

    with st.spinner("Reading schema..."):
        try:
            schema = fetch_schema_summary(engine_query, db=db_name)
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
            df = run_sql_to_df(engine_query, sql)
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
