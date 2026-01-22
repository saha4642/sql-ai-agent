# app.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

import mysql.connector
from mysql.connector import Error as MySQLError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from langchain_openai import ChatOpenAI

# ============================================================
# 1) URL / ENV HELPERS  (Fix: force mysql+mysqlconnector)
# ============================================================

SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _env(*keys: str, default: str = "") -> str:
    """Return first non-empty env var among keys."""
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def normalize_sqlalchemy_mysql_url(url: str) -> str:
    """
    Railway often provides: mysql://user:pass@host:port/db
    SQLAlchemy must use an explicit driver to avoid MySQLdb:
      mysql+mysqlconnector://user:pass@host:port/db
    """
    u = (url or "").strip()
    if not u:
        return u

    # If already explicit and not mysqldb, keep
    if u.startswith("mysql+mysqlconnector://"):
        return u

    # If it is mysqldb, replace
    if u.startswith("mysql+mysqldb://"):
        return "mysql+mysqlconnector://" + u[len("mysql+mysqldb://") :]

    # If no driver specified (mysql://), force mysqlconnector
    if u.startswith("mysql://"):
        return "mysql+mysqlconnector://" + u[len("mysql://") :]

    # If something else, try to keep it; but for mysql, still force
    if u.startswith("mysql+"):
        return u

    return u


def safe_int_port(port: str, default: int = 3306) -> int:
    try:
        return int(str(port).strip())
    except Exception:
        return default


@dataclass
class MySQLCfg:
    host: str
    port: int
    user: str
    password: str
    database: str


def parse_sqlalchemy_url_to_cfg(url_str: str) -> Tuple[str, MySQLCfg]:
    """
    Returns (normalized_sqlalchemy_url, cfg)
    """
    sa_url = normalize_sqlalchemy_mysql_url(url_str)
    u = make_url(sa_url)

    cfg = MySQLCfg(
        host=u.host or "localhost",
        port=u.port or 3306,
        user=u.username or "root",
        password=u.password or "",
        database=u.database or "railway",
    )
    return sa_url, cfg


def build_sqlalchemy_url(cfg: MySQLCfg) -> str:
    # quote credentials to be safe
    user = quote_plus(cfg.user)
    pw = quote_plus(cfg.password)
    return f"mysql+mysqlconnector://{user}:{pw}@{cfg.host}:{cfg.port}/{cfg.database}"


# ============================================================
# 2) APP CONFIG: How we decide ADMIN vs NLQ connections
# ============================================================

def get_admin_url_and_cfg() -> Tuple[str, MySQLCfg]:
    """
    Admin/root connection source order:
    1) DATABASE_URL_ROOT
    2) MYSQL_URL (Railway usually root)
    3) compose from MYSQL_ROOT_* + host/port/db vars
    """
    raw = _env("DATABASE_URL_ROOT", "MYSQL_URL")
    if raw:
        return parse_sqlalchemy_url_to_cfg(raw)

    # Compose fallback from separate vars
    host = _env("MYSQL_HOST", "MYSQLHOST", default="mysql.railway.internal")
    port = safe_int_port(_env("MYSQL_PORT", "MYSQLPORT", default="3306"))
    user = _env("MYSQL_ROOT_USER", default="root")
    pw = _env("MYSQL_ROOT_PASSWORD", default="")
    db = _env("MYSQL_DATABASE", "MYSQL_DB", "MYSQL_DEFAULT_DB", default="railway")

    cfg = MySQLCfg(host=host, port=port, user=user, password=pw, database=db)
    return build_sqlalchemy_url(cfg), cfg


def get_nlq_url_and_cfg(admin_cfg: MySQLCfg) -> Tuple[str, MySQLCfg]:
    """
    NLQ (read-only) connection source order:
    1) DATABASE_URL (if you set it to nlq_user)
    2) compose from NLQ_USER/NLQ_PASSWORD + admin host/port/db
    """
    raw = _env("DATABASE_URL")
    if raw:
        return parse_sqlalchemy_url_to_cfg(raw)

    nlq_user = _env("NLQ_USER", default="nlq_user")
    nlq_pass = _env("NLQ_PASSWORD", default="")

    cfg = MySQLCfg(
        host=admin_cfg.host,
        port=admin_cfg.port,
        user=nlq_user,
        password=nlq_pass,
        database=admin_cfg.database,
    )
    return build_sqlalchemy_url(cfg), cfg


@st.cache_resource
def get_engine(url: str) -> Engine:
    # pool_pre_ping helps on Railway
    return create_engine(normalize_sqlalchemy_mysql_url(url), pool_pre_ping=True)


def test_engine(engine: Engine) -> Optional[str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return None
    except Exception as e:
        return str(e)


# ============================================================
# 3) SQL SAFETY + LLM
# ============================================================

def get_llm() -> ChatOpenAI:
    api_key = _env("OPENAI_API_KEY")
    model = _env("OPENAI_MODEL", default="gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.0)


def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out or "")
    if m:
        return m.group(1).strip()
    return (text_out or "").strip()


def is_read_only_sql(sql: str) -> bool:
    s = (sql or "").strip()
    low = re.sub(r"\s+", " ", s.lower())
    if not (low.startswith("select") or low.startswith("with")):
        return False
    blocked = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
        "grant", "revoke", "commit", "rollback",
    ]
    return not any(re.search(rf"\b{k}\b", low) for k in blocked)


def normalize_sql(sql: str) -> str:
    s = (sql or "").strip()
    s = s.rstrip(";").strip()
    return s + ";"


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())

    if re.search(r"\blimit\s+\d+\b", low):
        return s

    # Add a default LIMIT for non-count SELECTs
    if low.startswith("select") and ("count(" not in low):
        return s.rstrip(";") + f" LIMIT {default_limit};"

    return s


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
        cols = ", ".join([f"{r.COLUMN_NAME}({r.DATA_TYPE})" for r in g.itertuples(index=False)])
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
        "- Use the provided schema.\n"
        "- If ambiguous, make a reasonable assumption.\n"
    )

    user = (
        f"SCHEMA:\n{schema}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Return ONLY SQL."
    )

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
# 4) NLQ USER PROVISION (CREATE USER + GRANT SELECT)
# ============================================================

def provision_nlq_user(admin_cfg: MySQLCfg, nlq_user: str, nlq_pass: str) -> None:
    """
    Creates nlq_user@'%' and grants SELECT on <db>.*
    Uses mysql.connector directly (admin/root credentials).
    """
    if not nlq_pass:
        raise RuntimeError("NLQ_PASSWORD is empty. Set NLQ_PASSWORD in Railway variables.")

    conn = mysql.connector.connect(
        host=admin_cfg.host,
        port=int(admin_cfg.port),
        user=admin_cfg.user,
        password=admin_cfg.password,
        database=admin_cfg.database,  # connect into db
    )
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE USER IF NOT EXISTS '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_pass,))
        cur.execute(f"GRANT SELECT ON `{admin_cfg.database}`.* TO '{nlq_user}'@'%';")
        cur.execute("FLUSH PRIVILEGES;")
        conn.commit()
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


# ============================================================
# 5) STREAMLIT UI
# ============================================================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

admin_url, admin_cfg = get_admin_url_and_cfg()
nlq_url, nlq_cfg = get_nlq_url_and_cfg(admin_cfg)

with st.sidebar:
    st.header("Railway / MySQL URLs (auto)")

    st.caption("Admin (root) URL source: DATABASE_URL_ROOT → MYSQL_URL → composed vars")
    st.code(normalize_sqlalchemy_mysql_url(admin_url), language="text")

    st.caption("NLQ (read-only) URL source: DATABASE_URL → (NLQ_USER/NLQ_PASSWORD + admin host/db)")
    st.code(nlq_url if "NLQ_PASSWORD" not in nlq_url else "configured", language="text")

    st.divider()
    st.header("NLQ User")
    nlq_user = st.text_input("NLQ_USER", value=_env("NLQ_USER", default="nlq_user"))
    nlq_pass = st.text_input("NLQ_PASSWORD", type="password", value=_env("NLQ_PASSWORD"))

    c1, c2 = st.columns(2)
    with c1:
        btn_test = st.button("Test Connections")
    with c2:
        btn_provision = st.button("Create/Grant NLQ")

    st.divider()
    st.header("OpenAI")
    st.caption("Set OPENAI_API_KEY and OPENAI_MODEL in Railway variables.")
    st.code(f"OPENAI_MODEL={_env('OPENAI_MODEL', default='gpt-4o-mini')}", language="text")

# ---- Actions: test & provision
if btn_test:
    admin_engine = get_engine(admin_url)
    nlq_engine = get_engine(nlq_url)

    err_admin = test_engine(admin_engine)
    err_nlq = test_engine(nlq_engine)

    if err_admin:
        st.sidebar.error(f"Admin connection failed: {err_admin}")
    else:
        st.sidebar.success("Admin connection OK ✅")

    if err_nlq:
        st.sidebar.error(f"NLQ connection failed: {err_nlq}")
    else:
        st.sidebar.success("NLQ connection OK ✅")

if btn_provision:
    try:
        # Use current sidebar values, but keep admin cfg from URL parse
        provision_nlq_user(admin_cfg, nlq_user=nlq_user, nlq_pass=nlq_pass)
        st.sidebar.success("NLQ user created + SELECT granted ✅")

        # Refresh NLQ URL with the new credentials
        nlq_cfg = MySQLCfg(
            host=admin_cfg.host,
            port=admin_cfg.port,
            user=nlq_user,
            password=nlq_pass,
            database=admin_cfg.database,
        )
        nlq_url = build_sqlalchemy_url(nlq_cfg)

    except Exception as e:
        st.sidebar.error(f"Provision NLQ failed: {e}")

# ============================================================
# 6) MAIN APP: Ask question -> SQL -> DataFrame (NLQ engine)
# ============================================================

st.subheader("Ask a question")
question = st.text_area(
    "Example: Which country's customers spent the most by invoice?",
    height=90,
)

run_btn = st.button("Run Query")

if run_btn:
    # Build engines
    admin_engine = get_engine(admin_url)
    nlq_engine = get_engine(nlq_url)

    # Check NLQ engine works
    err = test_engine(nlq_engine)
    if err:
        st.error(f"Database connection failed (NLQ user): {err}")
        st.info("Fix: Click **Create/Grant NLQ** in the sidebar, and ensure **NLQ_PASSWORD** is set.")
        st.stop()

    # LLM
    try:
        llm = get_llm()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Reading schema..."):
        schema = fetch_schema_summary(nlq_engine, db=nlq_cfg.database)

    with st.spinner("Generating SQL..."):
        sql = generate_sql_from_question(llm, schema, question)

    st.markdown("### Generated SQL")
    st.code(sql, language="sql")

    if not is_read_only_sql(sql):
        st.error("Blocked: generated SQL is not read-only (SELECT/WITH only).")
        st.stop()

    with st.spinner("Running query..."):
        try:
            df = run_sql_to_df(nlq_engine, sql)
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
        summary = llm.invoke(summary_prompt).content

    st.markdown("### Summary")
    st.write(summary)
