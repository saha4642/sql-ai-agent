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
# 0) Env helpers
# ============================================================

def env_any(*names: str) -> str:
    """
    Return the first non-empty env var among names.
    """
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v)
    return ""


# ============================================================
# 1) URL normalization (SQLAlchemy + mysql-connector)
# ============================================================

def normalize_mysql_sqlalchemy_url(raw: str) -> str:
    """
    Accepts common Railway/managed-MySQL URLs and returns a valid SQLAlchemy URL.

    Fixes:
      - mysql://... -> mysql+mysqlconnector://...
      - mysql+mysqldb://... -> mysql+mysqlconnector://...
      - strips accidental quotes/spaces
      - URL-encodes username/password (handles @ : / ? # etc.)
      - validates via SQLAlchemy make_url()

    Raises:
      ValueError if empty/None.
      sqlalchemy.exc.ArgumentError if the URL is structurally invalid.
    """
    if not raw or str(raw).strip() == "":
        raise ValueError("Database URL is empty/None. Check environment variables.")

    s = str(raw).strip().strip('"').strip("'")

    # Force mysql-connector driver
    if s.startswith("mysql://"):
        s = s.replace("mysql://", "mysql+mysqlconnector://", 1)
    elif s.startswith("mysql+mysqldb://"):
        s = s.replace("mysql+mysqldb://", "mysql+mysqlconnector://", 1)

    # Parse using SQLAlchemy
    u = make_url(s)

    # Encode creds to avoid URL parsing problems when they contain special chars
    username = quote_plus(u.username) if u.username else None
    password = quote_plus(u.password) if u.password else None
    u = u.set(username=username, password=password)

    return str(u)


def is_root_user(sqlalchemy_url: str) -> bool:
    try:
        u = make_url(sqlalchemy_url)
        return (u.username or "").lower() == "root"
    except Exception:
        return False


def assert_not_root(sqlalchemy_url: str, label: str = "DB URL") -> None:
    if is_root_user(sqlalchemy_url):
        raise RuntimeError(
            f"{label} is using user 'root'. Railway blocks root for app connections. "
            f"Use APP_MYSQL_USER/APP_MYSQL_PASSWORD (recommended) or non-root MYSQL_USER/MYSQL_PASSWORD."
        )


# ============================================================
# 2) Build URL from discrete vars (FIXED precedence)
# ============================================================

def build_url_from_discrete_vars() -> str:
    """
    Builds a mysql://user:pass@host:port/db URL using discrete env vars.

    CRITICAL FIX:
      - APP_MYSQL_USER / APP_MYSQL_PASSWORD ALWAYS win if present
      - If fallback resolves to root -> return "" (hard block)
      - Prefer MYSQL_DATABASE, then MYSQL_DB, then MYSQL_DEFAULT_DB
    """
    app_user = (env_any("APP_MYSQL_USER") or "").strip()
    app_pw = (env_any("APP_MYSQL_PASSWORD") or "").strip()

    rail_user = (env_any("MYSQL_USER", "MYSQLUSER") or "").strip()
    rail_pw = (env_any("MYSQL_PASSWORD", "MYSQLPASSWORD") or "").strip()

    if app_user and app_pw:
        user, pw = app_user, app_pw
    else:
        user, pw = rail_user, rail_pw

    # HARD BLOCK: never allow root from discrete vars
    if (user or "").lower() == "root":
        return ""

    host = (env_any("MYSQL_HOST", "MYSQLHOST") or "").strip()
    port = (env_any("MYSQL_PORT", "MYSQLPORT") or "").strip()

    db = (
        (env_any("MYSQL_DATABASE") or "").strip()
        or (env_any("MYSQL_DB") or "").strip()
        or (env_any("MYSQL_DEFAULT_DB") or "").strip()
    )

    if all([user, pw, host, port, db]):
        return f"mysql://{user}:{pw}@{host}:{port}/{db}"

    return ""


# ============================================================
# 3) SQLAlchemy engine helpers
# ============================================================

@st.cache_resource(show_spinner=False)
def get_engine(sqlalchemy_url: str) -> Engine:
    safe_url = normalize_mysql_sqlalchemy_url(sqlalchemy_url)
    return create_engine(safe_url, pool_pre_ping=True, pool_recycle=1800)


def test_engine(engine: Engine) -> Tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ============================================================
# 4) SQL safety + LLM helpers
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
    """
    Strong read-only guard:
    - Must start with SELECT or WITH
    - Must not contain any write/ddl keywords
    """
    low = re.sub(r"\s+", " ", (sql or "").strip().lower())
    if not (low.startswith("select") or low.startswith("with")):
        return False

    blocked = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
        "grant", "revoke", "commit", "rollback", "call", "load data", "outfile"
    ]
    return not any(re.search(rf"\b{re.escape(k)}\b", low) for k in blocked)


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    """
    Adds LIMIT only if there isn't one already and it isn't a pure COUNT query.
    Avoids duplicate LIMIT like: LIMIT 200 LIMIT 200
    """
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())

    if re.search(r"\blimit\s+\d+\b", low):
        return s

    if low.startswith("select") and ("count(" not in low):
        return s.rstrip(";") + f" LIMIT {default_limit};"

    return s


def get_llm(api_key: str, model: str, temperature: float = 0.0):
    # Import here so the app can still boot even if OpenAI deps are misconfigured
    from langchain_openai import ChatOpenAI
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


def generate_sql_from_question(llm, schema: str, question: str) -> str:
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
# 5) Streamlit UI
# ============================================================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

# Sidebar: OpenAI
with st.sidebar:
    st.header("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=env_any("OPENAI_API_KEY"))
    openai_model = st.text_input("OPENAI_MODEL", value=env_any("OPENAI_MODEL") or "gpt-4o-mini")

    st.divider()
    st.header("MySQL (Railway)")
    st.caption(
        "This app uses discrete Railway vars (MYSQL_HOST/PORT/DATABASE) and "
        "prefers APP_MYSQL_USER/APP_MYSQL_PASSWORD. Root is blocked."
    )

    btn_clear = st.button("Clear DB Cache")
    btn_test = st.button("Test Connection")

    st.divider()
    st.header("DB Debug (safe)")
    # show whether app creds are present (no secrets)
    st.write("APP_MYSQL_USER set:", bool(env_any("APP_MYSQL_USER")))
    st.write("APP_MYSQL_PASSWORD set:", bool(env_any("APP_MYSQL_PASSWORD")))
    st.write("MYSQL_USER present:", bool(env_any("MYSQL_USER", "MYSQLUSER")))
    st.write("MYSQL_DATABASE present:", bool(env_any("MYSQL_DATABASE", "MYSQL_DB", "MYSQL_DEFAULT_DB")))

if btn_clear:
    get_engine.clear()
    st.sidebar.success("DB cache cleared ✅")

# ------------------------------------------------------------
# Build DB URL (DISCRETE VARS ONLY) + hard block root
# ------------------------------------------------------------
raw_url = build_url_from_discrete_vars()
if not raw_url:
    st.error(
        "DB credentials invalid or root-only.\n\n"
        "Fix:\n"
        "- Ensure APP_MYSQL_USER and APP_MYSQL_PASSWORD are set to your non-root user (e.g., app_ro)\n"
        "- Ensure MYSQL_HOST, MYSQL_PORT, and MYSQL_DATABASE are set\n"
        "- If MYSQL_USER is 'root', it will be blocked unless APP_MYSQL_* is provided."
    )
    st.stop()

# Normalize for SQLAlchemy + verify not root
try:
    sqlalchemy_url = normalize_mysql_sqlalchemy_url(raw_url)
    assert_not_root(sqlalchemy_url, label="Admin/app URL")
except Exception as e:
    st.error(str(e))
    st.stop()

# Parse URL to get DB name
try:
    parsed = make_url(sqlalchemy_url)
    db_name = parsed.database or env_any("MYSQL_DATABASE") or env_any("MYSQL_DB") or env_any("MYSQL_DEFAULT_DB") or "railway"
except Exception:
    st.error("Could not parse DB name from URL.")
    st.stop()

# Engine
engine = get_engine(sqlalchemy_url)

# Tests
if btn_test:
    ok, msg = test_engine(engine)
    if ok:
        st.sidebar.success("DB connection OK ✅")
    else:
        st.sidebar.error(f"DB connection failed: {msg}")

# Connection status
with st.expander("Connection status", expanded=True):
    try:
        u_dbg = make_url(sqlalchemy_url)
        st.write(f"**Using user:** `{u_dbg.username}`")
        st.write(f"**Host:** `{u_dbg.host}`  **Port:** `{u_dbg.port}`  **DB:** `{db_name}`")
    except Exception:
        pass

    ok, msg = test_engine(engine)
    st.write(f"**Engine:** {'✅ OK' if ok else '❌ FAIL'}")
    if not ok:
        st.code(msg)

# ============================================================
# 6) Main: Ask question -> SQL -> DataFrame
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
