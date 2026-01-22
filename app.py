# app.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError

from langchain_openai import ChatOpenAI

# ============================================================
# 0) Critical Fix: Force SQLAlchemy to NOT use MySQLdb
# ============================================================

def normalize_mysql_url(url: str) -> str:
    """
    Railway usually provides: mysql://user:pass@host:port/db
    If we pass that to SQLAlchemy, it often tries MySQLdb (mysqlclient) -> ModuleNotFoundError: MySQLdb
    Fix: force mysql+mysqlconnector:// (uses mysql-connector-python).
    """
    u = (url or "").strip()
    if not u:
        return ""

    if u.startswith("mysql+mysqlconnector://"):
        return u

    if u.startswith("mysql+mysqldb://"):
        return "mysql+mysqlconnector://" + u[len("mysql+mysqldb://") :]

    if u.startswith("mysql://"):
        return "mysql+mysqlconnector://" + u[len("mysql://") :]

    # If already explicit driver (e.g., mysql+pymysql://), leave it
    return u


@st.cache_resource(show_spinner=False)
def get_engine(url: str) -> Engine:
    url = normalize_mysql_url(url)
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800)


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
# 2) Optional: Provision a read-only NLQ user (only if admin allows)
# ============================================================

NLQ_USER_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")

def provision_nlq_user(engine_admin: Engine, db_name: str, nlq_user: str, nlq_password: str) -> None:
    """
    Creates nlq_user@'%' and grants SELECT on db.*.
    NOTE: Some managed MySQL providers do NOT allow CREATE USER/GRANT from the app.
    In that case this will fail with 1045/1142 etc — then just use MYSQL_URL for querying.
    """
    if not NLQ_USER_RE.match(nlq_user or ""):
        raise ValueError("NLQ_USER must be alphanumeric/underscore (max 32 chars).")
    if not nlq_password:
        raise ValueError("NLQ_PASSWORD is empty.")

    # Identifiers can't be parameterized safely; validate + format.
    create_user_sql = f"CREATE USER IF NOT EXISTS `{nlq_user}`@'%' IDENTIFIED BY :pw;"
    alter_user_sql = f"ALTER USER `{nlq_user}`@'%' IDENTIFIED BY :pw;"
    grant_sql = f"GRANT SELECT ON `{db_name}`.* TO `{nlq_user}`@'%';"

    with engine_admin.connect() as conn:
        conn.execute(text(create_user_sql), {"pw": nlq_password})
        # Ensure password is set even if user already existed
        conn.execute(text(alter_user_sql), {"pw": nlq_password})
        conn.execute(text(grant_sql))
        conn.execute(text("FLUSH PRIVILEGES;"))
        conn.commit()


# ============================================================
# 3) Streamlit UI
# ============================================================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

# --- Defaults from env (Railway-friendly)
ENV_MYSQL_URL = os.getenv("MYSQL_URL", "")
ENV_DATABASE_URL = os.getenv("DATABASE_URL", "")
ENV_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ENV_NLQ_USER = os.getenv("NLQ_USER", "nlq_user")
ENV_NLQ_PASSWORD = os.getenv("NLQ_PASSWORD", "")

with st.sidebar:
    st.header("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_model = st.text_input("OPENAI_MODEL", value=ENV_OPENAI_MODEL)

    st.divider()
    st.header("MySQL URLs")
    st.caption("Use Railway’s MYSQL_URL. This app converts mysql:// → mysql+mysqlconnector:// automatically.")

    admin_url_input = st.text_input("MYSQL_URL (admin/app)", value=ENV_MYSQL_URL)
    # Optional separate read-only URL (if you created one)
    nlq_url_input = st.text_input("DATABASE_URL (read-only, optional)", value=ENV_DATABASE_URL)

    st.divider()
    st.header("Optional: Read-only NLQ user")
    use_nlq_user = st.checkbox("Use separate NLQ user for queries", value=False)
    nlq_user = st.text_input("NLQ_USER", value=ENV_NLQ_USER, disabled=not use_nlq_user)
    nlq_password = st.text_input("NLQ_PASSWORD", type="password", value=ENV_NLQ_PASSWORD, disabled=not use_nlq_user)

    cols = st.columns(2)
    btn_test = cols[0].button("Test Connections")
    btn_grant = cols[1].button("Create/Grant NLQ")

# --- Validate admin URL
admin_url = normalize_mysql_url(admin_url_input)
if not admin_url:
    st.error("MYSQL_URL is missing. Set it in Railway variables or paste it in the sidebar.")
    st.stop()

# Parse admin URL to get DB name (for schema + grants)
try:
    admin_parsed = make_url(admin_url)
    db_name = admin_parsed.database or (os.getenv("MYSQL_DATABASE") or "railway")
except Exception:
    st.error("Could not parse MYSQL_URL. Make sure it looks like mysql://user:pass@host:port/db")
    st.stop()

engine_admin = get_engine(admin_url)

# --- Decide which engine to use for querying
# Priority:
# 1) If DATABASE_URL provided (read-only url) -> use it
# 2) If "use NLQ user" checked -> build URL from admin host/port/db + NLQ creds
# 3) Else use admin/app MYSQL_URL directly
query_url = ""
if nlq_url_input.strip():
    query_url = normalize_mysql_url(nlq_url_input.strip())
elif use_nlq_user and nlq_user.strip() and nlq_password.strip():
    # Build derived NLQ URL from admin host/port/db
    host = admin_parsed.host
    port = admin_parsed.port or 3306
    query_url = f"mysql+mysqlconnector://{nlq_user}:{nlq_password}@{host}:{port}/{db_name}"
else:
    query_url = admin_url

engine_query = get_engine(query_url)

# --- Actions
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

if btn_grant:
    try:
        provision_nlq_user(engine_admin, db_name=db_name, nlq_user=nlq_user, nlq_password=nlq_password)
        st.sidebar.success(f"Granted SELECT to `{nlq_user}` on `{db_name}` ✅")
    except Exception as e:
        st.sidebar.error(f"Create/Grant NLQ failed: {e}")
        st.sidebar.info(
            "If your provider blocks CREATE USER/GRANT from apps, "
            "leave 'Use separate NLQ user' OFF and just query using MYSQL_URL."
        )

# --- Show status
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
# 4) Main: Ask question -> SQL -> DataFrame
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
