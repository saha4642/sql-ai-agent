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
    """Return the first non-empty env var among names."""
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v)
    return ""


# ============================================================
# 1) Build DB URL from discrete vars (APP_* always wins)
#    Uses PyMySQL driver to avoid caching_sha2 handshake issues.
# ============================================================

def build_raw_mysql_url_from_discrete_vars() -> str:
    """
    Build a mysql://user:pass@host:port/db URL from Railway discrete variables.

    Precedence:
      - APP_MYSQL_USER / APP_MYSQL_PASSWORD ALWAYS win if present
      - Else fallback to MYSQL_USER/MYSQL_PASSWORD (often root; root blocked)

    Requires:
      - MYSQL_HOST (or MYSQLHOST)
      - MYSQL_PORT (or MYSQLPORT)
      - MYSQL_DATABASE (or MYSQL_DB or MYSQL_DEFAULT_DB)
    """
    app_user = (env_any("APP_MYSQL_USER") or "").strip()
    app_pw = (env_any("APP_MYSQL_PASSWORD") or "").strip()

    rail_user = (env_any("MYSQL_USER", "MYSQLUSER") or "").strip()
    rail_pw = (env_any("MYSQL_PASSWORD", "MYSQLPASSWORD") or "").strip()

    if app_user and app_pw:
        user, pw = app_user, app_pw
    else:
        user, pw = rail_user, rail_pw

    # Railway blocks root for app connections
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


def normalize_mysql_sqlalchemy_url(raw: str) -> str:
    """
    Convert raw MySQL URL into a SQLAlchemy URL using PyMySQL driver:
      mysql://... -> mysql+pymysql://...

    Also:
      - strips quotes/spaces
      - URL-encodes username/password
      - validates via SQLAlchemy make_url()
    """
    if not raw or str(raw).strip() == "":
        raise ValueError("Database URL is empty. Check Railway variables.")

    s = str(raw).strip().strip('"').strip("'")

    # Force PyMySQL driver (important)
    if s.startswith("mysql+pymysql://"):
        pass
    elif s.startswith("mysql://"):
        s = s.replace("mysql://", "mysql+pymysql://", 1)
    else:
        # If some other mysql+driver:// is provided, normalize to pymysql
        s = re.sub(r"^mysql\+[^:]+://", "mysql+pymysql://", s, count=1)

    u = make_url(s)

    # Encode creds (handles special chars)
    username = quote_plus(u.username) if u.username else None
    password = quote_plus(u.password) if u.password else None
    u = u.set(username=username, password=password)

    return str(u)


def assert_not_root(sqlalchemy_url: str) -> None:
    u = make_url(sqlalchemy_url)
    if (u.username or "").lower() == "root":
        raise RuntimeError(
            "Admin/app URL is using user 'root'. Railway blocks root for app connections. "
            "Use APP_MYSQL_USER/APP_MYSQL_PASSWORD with a non-root user."
        )


# ============================================================
# 2) Engine helpers
# ============================================================

@st.cache_resource(show_spinner=False)
def get_engine(sqlalchemy_url: str) -> Engine:
    safe_url = normalize_mysql_sqlalchemy_url(sqlalchemy_url)
    assert_not_root(safe_url)
    # PyMySQL generally handles caching_sha2_password smoothly.
    return create_engine(safe_url, pool_pre_ping=True, pool_recycle=1800)


def test_engine(engine: Engine) -> Tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ============================================================
# 3) SQL safety + LLM helpers
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


def get_llm(api_key: str, model: str, temperature: float = 0.0):
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
# 4) Streamlit UI
# ============================================================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

with st.sidebar:
    st.header("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=env_any("OPENAI_API_KEY"))
    openai_model = st.text_input("OPENAI_MODEL", value=env_any("OPENAI_MODEL") or "gpt-4o-mini")

    st.divider()
    st.header("MySQL (Railway)")
    st.caption(
        "Uses Railway discrete vars (MYSQL_HOST/PORT/DATABASE). "
        "Prefers APP_MYSQL_USER/APP_MYSQL_PASSWORD. Root is blocked. "
        "Driver: PyMySQL."
    )

    btn_clear = st.button("Clear DB Cache")
    btn_test = st.button("Test Connection")

    st.divider()
    st.header("DB Debug (safe)")
    st.write("APP_MYSQL_USER set:", bool(env_any("APP_MYSQL_USER")))
    st.write("APP_MYSQL_PASSWORD set:", bool(env_any("APP_MYSQL_PASSWORD")))
    st.write("MYSQL_HOST set:", bool(env_any("MYSQL_HOST", "MYSQLHOST")))
    st.write("MYSQL_PORT set:", bool(env_any("MYSQL_PORT", "MYSQLPORT")))
    st.write("MYSQL_DATABASE set:", bool(env_any("MYSQL_DATABASE", "MYSQL_DB", "MYSQL_DEFAULT_DB")))
    st.write("MYSQL_USER present:", bool(env_any("MYSQL_USER", "MYSQLUSER")))

if btn_clear:
    get_engine.clear()
    st.sidebar.success("DB cache cleared ✅")

raw_url = build_raw_mysql_url_from_discrete_vars()
if not raw_url:
    st.error(
        "DB credentials invalid or root-only.\n\n"
        "Fix:\n"
        "- Ensure APP_MYSQL_USER and APP_MYSQL_PASSWORD are set to your non-root user\n"
        "- Ensure MYSQL_HOST, MYSQL_PORT, and MYSQL_DATABASE are set\n"
        "- If MYSQL_USER is 'root', it will be blocked unless APP_MYSQL_* is provided."
    )
    st.stop()

try:
    sqlalchemy_url = normalize_mysql_sqlalchemy_url(raw_url)
    parsed = make_url(sqlalchemy_url)
    db_name = parsed.database or env_any("MYSQL_DATABASE") or env_any("MYSQL_DB") or env_any("MYSQL_DEFAULT_DB") or "railway"
except Exception as e:
    st.error(f"Could not parse DB URL: {e}")
    st.stop()

engine = get_engine(sqlalchemy_url)

if btn_test:
    ok, msg = test_engine(engine)
    if ok:
        st.sidebar.success("DB connection OK ✅")
    else:
        st.sidebar.error(f"DB connection failed: {msg}")

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
