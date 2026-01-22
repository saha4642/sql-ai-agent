from __future__ import annotations

import os
import re
from typing import Optional, Tuple
from urllib.parse import quote_plus, urlsplit, unquote

import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error as MySQLError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

from langchain_openai import ChatOpenAI


# ----------------------------
# Regex / Guards
# ----------------------------
SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


# ----------------------------
# URL helpers (Railway-safe)
# ----------------------------
def normalize_sqlalchemy_mysql_url(url_str: str) -> str:
    """
    Accepts Railway-style mysql://user:pass@host:port/db
    Returns SQLAlchemy mysql+mysqlconnector://user:pass@host:port/db

    Handles special characters in passwords safely.
    """
    url_str = (url_str or "").strip()
    if not url_str:
        raise ValueError("Empty database URL")

    # If it's already a SQLAlchemy mysqlconnector URL, return it
    if url_str.startswith("mysql+mysqlconnector://"):
        return url_str

    # Some users store raw "mysql://..."
    if url_str.startswith("mysql://"):
        parts = urlsplit(url_str)
        user = unquote(parts.username or "")
        pwd = unquote(parts.password or "")
        host = parts.hostname or "localhost"
        port = parts.port or 3306
        db = (parts.path or "").lstrip("/")  # may be empty

        # Rebuild with mysqlconnector + quoted credentials
        user_q = quote_plus(user)
        pwd_q = quote_plus(pwd)
        if db:
            return f"mysql+mysqlconnector://{user_q}:{pwd_q}@{host}:{port}/{db}"
        return f"mysql+mysqlconnector://{user_q}:{pwd_q}@{host}:{port}"

    # As a fallback, try SQLAlchemy's parser (will raise if invalid)
    u = make_url(url_str)
    if u.drivername == "mysql":
        return str(u.set(drivername="mysql+mysqlconnector"))
    return url_str


def url_with_db(base_url: str, db: str) -> str:
    """
    Ensure the SQLAlchemy URL includes a database.
    If base_url has a db already, replace it.
    """
    u = make_url(base_url)
    return str(u.set(database=db))


# ----------------------------
# OpenAI / SQL helpers
# ----------------------------
def get_llm(api_key: str, model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)


def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out or "")
    if m:
        return m.group(1).strip()
    return (text_out or "").strip()


def normalize_sql(sql: str) -> str:
    s = (sql or "").strip().rstrip(";").strip()
    return s + ";"


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
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ).content

    sql = ensure_limit(normalize_sql(extract_sql(raw)), default_limit=200)
    return sql


def run_sql_to_df(engine: Engine, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)


# ----------------------------
# Admin ops (root)
# ----------------------------
def mysql_connect(host: str, port: int, user: str, password: str, database: Optional[str] = None):
    return mysql.connector.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
        database=database,
        autocommit=False,
    )


def create_database(host: str, port: int, user: str, password: str, db_name: str) -> None:
    conn = mysql_connect(host, port, user, password, database=None)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
    conn.commit()
    cur.close()
    conn.close()


def ensure_nlq_user(
    host: str,
    port: int,
    root_user: str,
    root_password: str,
    db_name: str,
    nlq_user: str,
    nlq_password: str,
) -> None:
    """
    Create or reset NLQ user and grant SELECT only.
    Uses wildcard host '%' because Railway service-to-service traffic does not come from localhost.
    """
    conn = mysql_connect(host, port, root_user, root_password, database=None)
    cur = conn.cursor()

    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")

    # Create user for any host
    cur.execute(f"CREATE USER IF NOT EXISTS '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_password,))
    # Reset password to be safe
    cur.execute(f"ALTER USER '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_password,))

    # Read-only grants
    cur.execute(f"GRANT SELECT ON `{db_name}`.* TO '{nlq_user}'@'%';")
    cur.execute("FLUSH PRIVILEGES;")

    conn.commit()
    cur.close()
    conn.close()


def import_sql_file(host: str, port: int, user: str, password: str, db_name: str, sql_text: str) -> int:
    """
    Import SQL into db_name using mysql-connector multi=True.
    Works for typical schema/data dumps (no DELIMITER/procedures).
    """
    conn = mysql_connect(host, port, user, password, database=db_name)
    cur = conn.cursor()
    ok = 0
    try:
        for _ in cur.execute(sql_text, multi=True):
            ok += 1
        conn.commit()
    except MySQLError:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
    return ok


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")


def read_env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v


with st.sidebar:
    st.header("OpenAI")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=read_env("OPENAI_API_KEY"))
    model = st.text_input("OPENAI_MODEL", value=read_env("OPENAI_MODEL", "gpt-4o-mini"))

    st.divider()
    st.header("Railway / MySQL URLs (recommended)")

    # Railway MySQL service provides MYSQL_URL (mysql://...)
    mysql_url = st.text_input("MYSQL_URL", value=read_env("MYSQL_URL", ""), help="Railway: ${MySQL.MYSQL_URL}")
    db_name = st.text_input("DB name", value=read_env("MYSQL_DEFAULT_DB", read_env("MYSQL_DATABASE", "railway")))

    st.caption("If MYSQL_URL is empty, we fallback to host/port/user/password fields below.")

    st.divider()
    st.header("Fallback: Manual Connection")
    mysql_host = st.text_input("MYSQL_HOST", value=read_env("MYSQL_HOST", read_env("MYSQLHOST", "mysql.railway.internal")))
    mysql_port = st.text_input("MYSQL_PORT", value=read_env("MYSQL_PORT", read_env("MYSQLPORT", "3306")))
    root_user = st.text_input("MYSQL_ROOT_USER", value=read_env("MYSQL_ROOT_USER", "root"))
    root_pass = st.text_input("MYSQL_ROOT_PASSWORD", type="password", value=read_env("MYSQL_ROOT_PASSWORD", read_env("MYSQLPASSWORD", "")))

    st.divider()
    st.header("NLQ (read-only) user")
    nlq_user = st.text_input("NLQ_USER", value=read_env("NLQ_USER", "nlq_user"))
    nlq_pass = st.text_input("NLQ_PASSWORD", type="password", value=read_env("NLQ_PASSWORD", ""))

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        btn_test = st.button("Test Connections")
    with c2:
        btn_provision = st.button("Create/Grant NLQ")
    with c3:
        btn_create_db = st.button("Create DB")

    st.divider()
    st.subheader("Optional: Import .sql into DB")
    sql_upload = st.file_uploader("Upload a .sql file", type=["sql"])
    btn_import = st.button("Import SQL")


def get_admin_connection_parts() -> Tuple[str, str, int, str, str]:
    """
    Returns (host, user, port, password, admin_sqlalchemy_url_with_db)
    """
    if mysql_url.strip():
        base = normalize_sqlalchemy_mysql_url(mysql_url.strip())
        # base may include db; replace with chosen db_name
        url_db = url_with_db(base, db_name)
        u = make_url(url_db)
        return u.host, u.username, int(u.port or 3306), u.password or "", url_db

    # Manual fallback
    host = mysql_host.strip()
    port = int(mysql_port.strip() or "3306")
    user = root_user.strip() or "root"
    pwd = root_pass
    url_db = f"mysql+mysqlconnector://{quote_plus(user)}:{quote_plus(pwd)}@{host}:{port}/{db_name}"
    return host, user, port, pwd, url_db


def get_nlq_sqlalchemy_url(admin_host: str, admin_port: int) -> str:
    """
    NLQ uses host/port from the MySQL service, but user/pass from NLQ vars.
    """
    user = (nlq_user or "").strip()
    pwd = nlq_pass or ""
    if not user:
        raise ValueError("NLQ_USER is empty")
    if not pwd:
        raise ValueError("NLQ_PASSWORD is empty")
    return f"mysql+mysqlconnector://{quote_plus(user)}:{quote_plus(pwd)}@{admin_host}:{admin_port}/{db_name}"


def test_engine(url: str) -> Optional[str]:
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return None
    except Exception as e:
        return str(e)


# Sidebar actions
admin_host, admin_user, admin_port, admin_pwd, admin_url_db = get_admin_connection_parts()

if btn_create_db:
    try:
        create_database(admin_host, admin_port, admin_user, admin_pwd, db_name)
        st.sidebar.success(f"Database `{db_name}` created (or already exists).")
    except Exception as e:
        st.sidebar.error(f"Create DB failed: {e}")

if btn_provision:
    try:
        if not nlq_pass.strip():
            st.sidebar.error("Set NLQ_PASSWORD first.")
        else:
            ensure_nlq_user(
                host=admin_host,
                port=admin_port,
                root_user=admin_user,
                root_password=admin_pwd,
                db_name=db_name,
                nlq_user=nlq_user.strip(),
                nlq_password=nlq_pass,
            )
            st.sidebar.success("NLQ user created/updated and granted SELECT ✅")
    except Exception as e:
        st.sidebar.error(f"Provision NLQ failed: {e}")

if btn_import:
    if not sql_upload:
        st.sidebar.warning("Upload a .sql file first.")
    else:
        try:
            sql_text = sql_upload.getvalue().decode("utf-8", errors="replace")
            # Ensure DB exists then import into it
            create_database(admin_host, admin_port, admin_user, admin_pwd, db_name)
            ok = import_sql_file(admin_host, admin_port, admin_user, admin_pwd, db_name, sql_text)
            st.sidebar.success(f"Import finished. Statements executed: {ok}.")
        except Exception as e:
            st.sidebar.error(f"Import failed: {e}")
            st.sidebar.info("If your .sql uses DELIMITER/stored procedures, import using MySQL Workbench or mysql CLI instead.")

if btn_test:
    admin_err = test_engine(admin_url_db)
    if admin_err:
        st.sidebar.error(f"Admin connection failed: {admin_err}")
    else:
        st.sidebar.success("Admin connection OK ✅")

    try:
        nlq_url = get_nlq_sqlalchemy_url(admin_host, admin_port)
        nlq_err = test_engine(nlq_url)
        if nlq_err:
            st.sidebar.error(f"NLQ connection failed: {nlq_err}")
        else:
            st.sidebar.success("NLQ connection OK ✅")
    except Exception as e:
        st.sidebar.error(f"NLQ config error: {e}")


# ----------------------------
# Main: Ask question -> SQL -> DataFrame
# ----------------------------
st.subheader("Ask a question")
question = st.text_area(
    "Example: Which country’s customers spent the most by invoice?",
    height=90,
)

run_btn = st.button("Run Query")

if run_btn:
    if not api_key.strip():
        st.error("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

    # NLQ engine first (read-only user)
    try:
        nlq_url = get_nlq_sqlalchemy_url(admin_host, admin_port)
        nlq_engine = create_engine(nlq_url, pool_pre_ping=True)
        with nlq_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        st.error(f"Database connection failed (NLQ user): {e}")
        st.info("Fix: Click **Create/Grant NLQ** in the sidebar, and ensure NLQ_PASSWORD is set.")
        st.stop()

    llm = get_llm(api_key=api_key, model=model, temperature=0.0)

    with st.spinner("Reading schema..."):
        schema = fetch_schema_summary(nlq_engine, db_name=db_name)

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
