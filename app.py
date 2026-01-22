from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error as MySQLError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url, URL

from langchain_openai import ChatOpenAI

# ----------------------------
# Regex / Guards
# ----------------------------
SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


# ----------------------------
# URL / Config helpers
# ----------------------------
@dataclass
class DbCfg:
    drivername: str
    username: str
    password: str
    host: str
    port: int
    database: str


def _sanitize_db_name(name: str) -> str:
    # allow letters, numbers, underscore only
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", (name or "").strip())
    return cleaned or "railway"


def to_sqlalchemy_mysqlconnector(url_str: str) -> str:
    """
    Railway often gives mysql://user:pass@host:port/db
    SQLAlchemy with mysql-connector expects mysql+mysqlconnector://...
    """
    if not url_str or not isinstance(url_str, str):
        raise ValueError("Empty database URL")

    u = make_url(url_str.strip())

    # Force mysql+mysqlconnector
    if u.drivername.startswith("mysql"):
        drivername = "mysql+mysqlconnector"
    else:
        # If somehow it's not mysql, keep it (but likely wrong for this app)
        drivername = u.drivername

    fixed = URL.create(
        drivername=drivername,
        username=u.username,
        password=u.password,
        host=u.host,
        port=u.port,
        database=u.database,
        query=dict(u.query) if u.query else None,
    )
    return str(fixed)


def parse_mysql_cfg(url_str: str) -> DbCfg:
    u = make_url(url_str.strip())
    if not u.drivername.startswith("mysql"):
        raise ValueError("URL is not a MySQL URL")
    if not u.host or not u.port or not u.username:
        raise ValueError("MySQL URL missing host/port/user")
    return DbCfg(
        drivername=u.drivername,
        username=u.username or "",
        password=u.password or "",
        host=u.host or "",
        port=int(u.port),
        database=u.database or "",
    )


def build_admin_url_and_cfg() -> Tuple[str, DbCfg]:
    """
    Prefer DATABASE_URL_ROOT if provided.
    Else use MYSQL_URL / DATABASE_URL.
    """
    raw = (
        os.getenv("DATABASE_URL_ROOT", "").strip()
        or os.getenv("MYSQL_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
    )
    if not raw:
        raise ValueError("Missing DATABASE_URL_ROOT / MYSQL_URL / DATABASE_URL in environment variables")

    # Ensure a database exists in URL string (Railway usually uses /railway)
    cfg = parse_mysql_cfg(raw)
    sa_url = to_sqlalchemy_mysqlconnector(raw)
    return sa_url, cfg


def build_nlq_url(db_name: str) -> str:
    """
    Build NLQ SQLAlchemy URL using same host/port but nlq_user creds.
    """
    admin_raw = (
        os.getenv("MYSQL_URL", "").strip()
        or os.getenv("DATABASE_URL", "").strip()
        or os.getenv("DATABASE_URL_ROOT", "").strip()
    )
    if not admin_raw:
        raise ValueError("Missing MYSQL_URL/DATABASE_URL for building NLQ connection")

    base = parse_mysql_cfg(admin_raw)

    nlq_user = os.getenv("NLQ_USER", "nlq_user").strip()
    nlq_pass = os.getenv("NLQ_PASSWORD", "").strip()
    if not nlq_pass:
        raise ValueError("NLQ_PASSWORD is missing")

    nlq_url = URL.create(
        drivername="mysql+mysqlconnector",
        username=nlq_user,
        password=nlq_pass,
        host=base.host,
        port=base.port,
        database=db_name,
    )
    return str(nlq_url)


# ----------------------------
# MySQL provisioning helpers
# ----------------------------
def admin_mysql_connect(cfg: DbCfg):
    return mysql.connector.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.username,
        password=cfg.password,
        autocommit=True,
    )


def create_database_if_needed(cfg: DbCfg, db_name: str) -> None:
    conn = admin_mysql_connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
        cur.close()
    finally:
        conn.close()


def ensure_nlq_user_and_grants(cfg: DbCfg, db_name: str, nlq_user: str, nlq_pass: str) -> None:
    """
    Creates or updates nlq_user and grants SELECT only on db_name.
    Works on MySQL 8+.
    """
    conn = admin_mysql_connect(cfg)
    try:
        cur = conn.cursor()

        # Create user if not exists, else update password
        cur.execute(f"CREATE USER IF NOT EXISTS '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_pass,))
        cur.execute(f"ALTER USER '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_pass,))

        # Grant read-only
        cur.execute(f"GRANT SELECT ON `{db_name}`.* TO '{nlq_user}'@'%';")
        cur.execute("FLUSH PRIVILEGES;")

        cur.close()
    finally:
        conn.close()


# ----------------------------
# LLM helpers
# ----------------------------
def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.0)


def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out)
    return (m.group(1).strip() if m else text_out.strip())


def normalize_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    return s + ";"


def is_read_only_sql(sql: str) -> bool:
    s = sql.strip()
    low = re.sub(r"\s+", " ", s.lower())
    if not (low.startswith("select") or low.startswith("with")):
        return False
    blocked = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
        "grant", "revoke", "commit", "rollback",
    ]
    return not any(re.search(rf"\b{k}\b", low) for k in blocked)


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())
    if re.search(r"\blimit\s+\d+\b", low):
        return s
    if low.startswith("select") and ("count(" not in low):
        return s.rstrip(";") + f" LIMIT {default_limit};"
    return s


def fetch_schema_summary(engine: Engine, db_name: str, max_tables: int = 200) -> str:
    q = text("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    df = pd.read_sql(q, engine, params={"db": db_name})
    if df.empty:
        return "(No tables found in this database yet.)"

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
        "- Return ONLY the SQL query.\n"
        "- Must be read-only (SELECT or WITH only).\n"
        "- Use the provided schema.\n"
    )
    user = f"SCHEMA:\n{schema}\n\nQUESTION:\n{question}\n\nReturn ONLY SQL."
    raw = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content
    sql = ensure_limit(normalize_sql(extract_sql(raw)), 200)
    return sql


def run_sql_to_df(engine: Engine, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

with st.sidebar:
    st.header("Database (Railway)")

    db_name = st.text_input(
        "Target database name",
        value=os.getenv("MYSQL_DEFAULT_DB", "chinook"),
        help="This is the DB your queries will run against.",
    )
    db_name = _sanitize_db_name(db_name)

    st.divider()
    st.subheader("Provision read-only user (NLQ)")
    nlq_user = st.text_input("NLQ_USER", value=os.getenv("NLQ_USER", "nlq_user"))
    nlq_pass = st.text_input("NLQ_PASSWORD", type="password", value=os.getenv("NLQ_PASSWORD", ""))

    c1, c2 = st.columns(2)
    with c1:
        test_btn = st.button("Test Admin + NLQ")
    with c2:
        grant_btn = st.button("Create/Grant NLQ")

    st.divider()
    st.caption("Tip: Ensure MYSQL_URL/DATABASE_URL points to Railway MySQL, and NLQ_PASSWORD is set.")


def test_connection_sqlalchemy(sa_url: str) -> Optional[str]:
    try:
        eng = create_engine(sa_url, pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return None
    except Exception as e:
        return str(e)


# --- Actions
if test_btn:
    try:
        admin_url, admin_cfg = build_admin_url_and_cfg()
        admin_err = test_connection_sqlalchemy(admin_url)

        if admin_err:
            st.sidebar.error(f"Admin connection failed: {admin_err}")
        else:
            st.sidebar.success("Admin connection OK ✅")

        if nlq_pass.strip():
            nlq_url = build_nlq_url(db_name)
            nlq_err = test_connection_sqlalchemy(nlq_url)
            if nlq_err:
                st.sidebar.error(f"NLQ connection failed: {nlq_err}")
            else:
                st.sidebar.success("NLQ connection OK ✅")
        else:
            st.sidebar.warning("NLQ_PASSWORD is empty, skipping NLQ test.")

    except Exception as e:
        st.sidebar.error(str(e))

if grant_btn:
    try:
        if not nlq_pass.strip():
            st.sidebar.error("Set NLQ_PASSWORD first.")
        else:
            admin_url, admin_cfg = build_admin_url_and_cfg()
            # create db
            create_database_if_needed(admin_cfg, db_name)
            # create/grant
            ensure_nlq_user_and_grants(admin_cfg, db_name, nlq_user.strip(), nlq_pass.strip())
            st.sidebar.success(f"NLQ user '{nlq_user}' created/updated and granted SELECT on `{db_name}` ✅")
    except Exception as e:
        st.sidebar.error(f"Provision NLQ failed: {e}")


# ----------------------------
# Main: Ask question -> SQL -> DataFrame
# ----------------------------
st.subheader("Ask a question")
question = st.text_area("Example: Top 10 customers by total spend?", height=90)
run_btn = st.button("Run Query")

if run_btn:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Build NLQ engine
    try:
        nlq_url = build_nlq_url(db_name)
        nlq_err = test_connection_sqlalchemy(nlq_url)
        if nlq_err:
            st.error(f"Database connection failed (NLQ user): {nlq_err}")
            st.info("Fix: Click Create/Grant NLQ in the sidebar, and ensure NLQ_PASSWORD is set.")
            st.stop()
        engine = create_engine(nlq_url, pool_pre_ping=True)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # LLM
    try:
        llm = get_llm()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Reading schema..."):
        schema = fetch_schema_summary(engine, db_name)

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

    st.markdown("### Results")
    st.dataframe(df, use_container_width=True)

    with st.spinner("Summarizing results..."):
        preview = df.head(20).to_csv(index=False)
        summary_prompt = (
            "Summarize the result for a business user in 2-5 bullet points.\n\n"
            f"Question: {question}\n\n"
            f"SQL:\n{sql}\n\n"
            f"CSV Preview:\n{preview}"
        )
        summary = llm.invoke(summary_prompt).content

    st.markdown("### Summary")
    st.write(summary)
