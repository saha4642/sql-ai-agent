from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error as MySQLError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL

from langchain_openai import ChatOpenAI


# ============================
# Helpers: env + safety guards
# ============================

SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def env_get(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty env var from names."""
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return v
    return default

def to_int(v: str, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default

def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out or "")
    if m:
        return m.group(1).strip()
    return (text_out or "").strip()

def normalize_sql(sql: str) -> str:
    s = (sql or "").strip()
    s = s.rstrip(";").strip()
    return s + ";"

def is_read_only_sql(sql: str) -> bool:
    """
    Strong read-only guard:
    - Must start with SELECT or WITH
    - Must not contain write/ddl keywords
    """
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
    """
    Adds LIMIT <default_limit> only if:
    - It's a SELECT query
    - AND there is no existing LIMIT
    - AND it's not a COUNT aggregate query
    """
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())

    if re.search(r"\blimit\s+\d+\b", low):
        return s

    if low.startswith("select") and ("count(" not in low):
        return s.rstrip(";") + f" LIMIT {default_limit};"

    return s


# ============================
# LLM
# ============================

def get_llm(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)

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


# ============================
# Database: SQLAlchemy + mysql-connector
# ============================

def build_engine(user: str, password: str, host: str, port: int, db: str) -> Engine:
    # URL.create safely escapes user/pass
    url = URL.create(
        drivername="mysql+mysqlconnector",
        username=user,
        password=password,
        host=host,
        port=port,
        database=db,
    )
    return create_engine(url, pool_pre_ping=True)

def fetch_schema_summary(engine: Engine, max_tables: int = 200) -> str:
    q = text("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    db = engine.url.database
    df = pd.read_sql(q, engine, params={"db": db})

    lines = []
    for tname, g in df.groupby("TABLE_NAME"):
        cols = ", ".join([f"{r.COLUMN_NAME}({r.DATA_TYPE})" for r in g.itertuples(index=False)])
        lines.append(f"- {tname}: {cols}")
        if len(lines) >= max_tables:
            lines.append("... (schema truncated)")
            break
    return "\n".join(lines)

def run_sql_to_df(engine: Engine, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)

def test_connection_mysql(host: str, port: int, user: str, password: str, db: Optional[str] = None) -> None:
    conn = mysql.connector.connect(host=host, port=port, user=user, password=password, database=db)
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    cur.fetchall()
    cur.close()
    conn.close()

def create_database_mysql(host: str, port: int, user: str, password: str, db_name: str) -> None:
    conn = mysql.connector.connect(host=host, port=port, user=user, password=password)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
    conn.commit()
    cur.close()
    conn.close()

def ensure_nlq_user_and_grants(
    host: str,
    port: int,
    root_user: str,
    root_password: str,
    db_name: str,
    nlq_user: str,
    nlq_password: str,
) -> None:
    """
    Creates nlq_user if missing and grants SELECT on db_name.*.
    Uses root credentials.
    """
    conn = mysql.connector.connect(host=host, port=port, user=root_user, password=root_password)
    cur = conn.cursor()

    # Create user if not exists
    cur.execute(f"CREATE USER IF NOT EXISTS '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_password,))
    # Grant only SELECT
    cur.execute(f"GRANT SELECT ON `{db_name}`.* TO '{nlq_user}'@'%';")
    cur.execute("FLUSH PRIVILEGES;")
    conn.commit()

    cur.close()
    conn.close()

def _split_sql_statements_basic(sql_text: str) -> list[str]:
    """
    Very simple SQL splitter.
    Works for most sample datasets (Chinook, many Sakila dumps) that don't use DELIMITER/procedures.
    If your file uses procedures/triggers/DELIMITER, import via mysqlsh SOURCE instead.
    """
    # Remove BOM and normalize newlines
    s = (sql_text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")

    # Strip common full-line comments
    lines = []
    for line in s.split("\n"):
        stripped = line.strip()
        if stripped.startswith("--") or stripped.startswith("#"):
            continue
        lines.append(line)
    s = "\n".join(lines)

    # Split on semicolons (basic)
    parts = [p.strip() for p in s.split(";")]
    stmts = []
    for p in parts:
        if p.strip():
            stmts.append(p.strip() + ";")
    return stmts

def import_sql_file_mysql(
    host: str,
    port: int,
    user: str,
    password: str,
    db_name: str,
    sql_text: str
) -> Tuple[int, int]:
    """
    Import .sql into db_name.

    1) Try mysql-connector multi=True (fast, if supported)
    2) If 'multi' not supported, fallback to basic statement-by-statement

    Returns (ok_statements, failed_statements)
    """
    conn = mysql.connector.connect(
        host=host, port=port, user=user, password=password, database=db_name, autocommit=False
    )
    cur = conn.cursor()

    ok = 0
    fail = 0

    try:
        # ---- Fast path: multi=True (if supported by this cursor build)
        try:
            for _ in cur.execute(sql_text, multi=True):  # type: ignore
                ok += 1
            conn.commit()
            return ok, fail
        except TypeError as te:
            # e.g., "unexpected keyword argument 'multi'"
            conn.rollback()
            # Fall back to basic splitting
            stmts = _split_sql_statements_basic(sql_text)
            for stmt in stmts:
                try:
                    cur.execute(stmt)
                    ok += 1
                except MySQLError:
                    fail += 1
                    # keep going; you can also choose to stop on first error
            conn.commit()
            return ok, fail

    except MySQLError:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — Upload DB + Query + DataFrame")


# --- Load defaults (Railway + local)
default_openai_key = env_get("OPENAI_API_KEY", default="")
default_openai_model = env_get("OPENAI_MODEL", default="gpt-4o-mini")

default_mysql_host = env_get("MYSQL_HOST", default="127.0.0.1")
default_mysql_port = to_int(env_get("MYSQL_PORT", default="3306"), 3306)

# Support both styles:
default_root_user = env_get("MYSQL_ROOT_USER", "MYSQL_USER", "MYSQL_USERNAME", default="root")
default_root_pass = env_get("MYSQL_ROOT_PASSWORD", "MYSQL_PASSWORD", default="")

default_db_name = env_get("MYSQL_DEFAULT_DB", "MYSQL_DB", "MYSQL_DATABASE", default="Chinook")

default_nlq_user = env_get("NLQ_USER", default="nlq_user")
default_nlq_pass = env_get("NLQ_PASSWORD", default="NLQpass123!")


with st.sidebar:
    st.header("OpenAI")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=default_openai_key)
    model = st.text_input("Model", value=default_openai_model)

    st.divider()
    st.header("MySQL (Admin / Root) — used for import + grants")
    mysql_host = st.text_input("Host", value=default_mysql_host)
    mysql_port_str = st.text_input("Port", value=str(default_mysql_port))
    mysql_port = to_int(mysql_port_str, default_mysql_port)

    root_user = st.text_input("Root/Admin User", value=default_root_user)
    root_pass = st.text_input("Root/Admin Password", type="password", value=default_root_pass)

    st.divider()
    st.header("Database")
    db_name = st.text_input("Database name", value=default_db_name)

    st.divider()
    st.header("Query User (Created Automatically)")
    nlq_user = st.text_input("NLQ user", value=default_nlq_user)
    nlq_pass = st.text_input("NLQ password", type="password", value=default_nlq_pass)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        do_test_root = st.button("Test Root Conn")
    with c2:
        do_test_nlq = st.button("Test NLQ Conn")

    st.divider()
    st.subheader("Upload & Import .sql")
    st.caption("This will: Create DB → Import SQL → Create nlq_user → Grant SELECT")
    sql_upload = st.file_uploader("Upload a .sql file", type=["sql"])
    do_import_all = st.button("Import SQL File (All Steps)")


# --- Sidebar actions

if do_test_root:
    try:
        test_connection_mysql(mysql_host, mysql_port, root_user, root_pass, None)
        st.sidebar.success("Root/Admin connection OK ✅")
    except Exception as e:
        st.sidebar.error(f"Root/Admin connection failed: {e}")

if do_test_nlq:
    try:
        test_connection_mysql(mysql_host, mysql_port, nlq_user, nlq_pass, db_name)
        st.sidebar.success("NLQ user connection OK ✅")
    except Exception as e:
        st.sidebar.error(f"NLQ user connection failed: {e}")

if do_import_all:
    if not sql_upload:
        st.sidebar.warning("Upload a .sql file first.")
    else:
        try:
            sql_text = sql_upload.getvalue().decode("utf-8", errors="replace")

            # 1) Create DB
            create_database_mysql(mysql_host, mysql_port, root_user, root_pass, db_name)

            # 2) Import SQL (as root/admin)
            ok, failed = import_sql_file_mysql(mysql_host, mysql_port, root_user, root_pass, db_name, sql_text)

            # 3) Create nlq user + grant SELECT
            ensure_nlq_user_and_grants(
                host=mysql_host,
                port=mysql_port,
                root_user=root_user,
                root_password=root_pass,
                db_name=db_name,
                nlq_user=nlq_user,
                nlq_password=nlq_pass,
            )

            st.sidebar.success(f"Import completed ✅  OK statements: {ok} | Failed: {failed}")
            if failed > 0:
                st.sidebar.warning(
                    "Some statements failed. If your SQL file contains procedures/triggers/DELIMITER, "
                    "import using MySQL Shell/CLI SOURCE instead."
                )

        except Exception as e:
            st.sidebar.error(f"Import failed ❌ {e}")
            st.sidebar.info(
                "Tip: If the .sql uses DELIMITER/procedures/triggers, import using MySQL Shell SOURCE instead."
            )


# ============================
# Main: Ask question → SQL → DataFrame
# ============================

st.subheader("Ask a question")
question = st.text_area(
    "Example: Which country's customers spent the most by invoice?",
    height=90,
)

run_btn = st.button("Run Query")


if run_btn:
    if not api_key.strip():
        st.error("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

    if not question.strip():
        st.error("Please type a question.")
        st.stop()

    # Connect using the NLQ user (SELECT-only)
    try:
        engine = build_engine(nlq_user, nlq_pass, mysql_host, mysql_port, db_name)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        st.error(f"Database connection failed (NLQ user): {e}")
        st.info("Make sure you imported the DB using the 'Import SQL File (All Steps)' button.")
        st.stop()

    llm = get_llm(api_key=api_key, model=model, temperature=0.0)

    with st.spinner("Reading schema..."):
        schema = fetch_schema_summary(engine)

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
        summary = llm.invoke(summary_prompt).content

    st.markdown("### Summary")
    st.write(summary)
