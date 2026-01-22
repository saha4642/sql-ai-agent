from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error as MySQLError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_openai import ChatOpenAI

# ============================================================
# Helpers
# ============================================================

SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Read-only guard (blocks any DDL/DML)
BLOCKED_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
    "grant", "revoke", "commit", "rollback", "call", "delimiter", "load", "outfile",
]

def parse_port(port_value: str, default: int = 3306) -> int:
    try:
        return int(str(port_value).strip())
    except Exception:
        return default

def safe_env(key: str, default: str = "") -> str:
    v = os.getenv(key, "")
    return v if v.strip() else default

def build_mysql_uri(user: str, password: str, host: str, port: int, db: str) -> str:
    # URL-encoding passwords can be necessary if it contains special chars.
    # For simplicity, keep as-is; if you hit issues, replace with urllib.parse.quote_plus(password).
    return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db}"

def get_llm(api_key: str, model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)

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
    s = (sql or "").strip()
    low = re.sub(r"\s+", " ", s.lower())

    if not (low.startswith("select") or low.startswith("with")):
        return False

    return not any(re.search(rf"\b{k}\b", low) for k in BLOCKED_KEYWORDS)

def ensure_limit(sql: str, default_limit: int = 200) -> str:
    """
    Adds LIMIT if missing and query isn't a pure COUNT aggregate.
    Prevents duplicate LIMIT.
    """
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())

    if re.search(r"\blimit\s+\d+\b", low):
        return s

    # don't limit a pure count query
    if low.startswith("select") and ("count(" not in low):
        return s.rstrip(";") + f" LIMIT {default_limit};"

    return s

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
# SQL Import (no cursor.execute(..., multi=True) to avoid errors)
# ============================================================

def has_delimiter(sql_text: str) -> bool:
    return bool(re.search(r"(?im)^\s*delimiter\s+", sql_text or ""))

def strip_sql_comments(sql_text: str) -> str:
    """
    Remove common SQL comments:
    - -- ...
    - # ...
    - /* ... */
    Keep it simple (good enough for typical sample DB scripts).
    """
    s = sql_text or ""
    # remove /* ... */
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    # remove -- ... and # ...
    s = re.sub(r"(?m)^\s*--.*?$", "", s)
    s = re.sub(r"(?m)^\s*#.*?$", "", s)
    return s

def split_sql_statements(sql_text: str) -> List[str]:
    """
    Split statements by semicolon; respects simple quoted strings.
    Not perfect for every SQL file, but works for most sample DB dumps.
    """
    s = strip_sql_comments(sql_text)
    s = s.replace("\r\n", "\n")

    stmts: List[str] = []
    buff: List[str] = []
    in_single = False
    in_double = False

    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_double:
            # toggle single unless escaped
            prev = s[i - 1] if i > 0 else ""
            if prev != "\\":
                in_single = not in_single
        elif ch == '"' and not in_single:
            prev = s[i - 1] if i > 0 else ""
            if prev != "\\":
                in_double = not in_double

        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buff).strip()
            if stmt:
                stmts.append(stmt)
            buff = []
        else:
            buff.append(ch)

        i += 1

    tail = "".join(buff).strip()
    if tail:
        stmts.append(tail)

    return stmts

def mysql_admin_connect(host: str, port: int, user: str, password: str, database: Optional[str] = None):
    kwargs = dict(host=host, port=port, user=user, password=password)
    if database:
        kwargs["database"] = database
    return mysql.connector.connect(**kwargs)

def create_database_if_needed(host: str, port: int, admin_user: str, admin_pass: str, db_name: str) -> None:
    conn = mysql_admin_connect(host, port, admin_user, admin_pass)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
    conn.commit()
    cur.close()
    conn.close()

def ensure_nlq_user_and_grants(
    host: str,
    port: int,
    admin_user: str,
    admin_pass: str,
    nlq_user: str,
    nlq_pass: str,
    db_name: str,
) -> None:
    """
    Create nlq user and grant read-only privileges on db_name.*.
    We grant for both '%' and 'localhost' to cover local and hosted use.
    """
    conn = mysql_admin_connect(host, port, admin_user, admin_pass)
    cur = conn.cursor()

    # Create user for '%' (common for hosted) and localhost (local dev)
    cur.execute(f"CREATE USER IF NOT EXISTS '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_pass,))
    cur.execute(f"CREATE USER IF NOT EXISTS '{nlq_user}'@'localhost' IDENTIFIED BY %s;", (nlq_pass,))

    # Read-only grants
    grant_sql = f"GRANT SELECT, SHOW VIEW ON `{db_name}`.* TO '{nlq_user}'@'%';"
    grant_sql2 = f"GRANT SELECT, SHOW VIEW ON `{db_name}`.* TO '{nlq_user}'@'localhost';"
    cur.execute(grant_sql)
    cur.execute(grant_sql2)
    cur.execute("FLUSH PRIVILEGES;")

    conn.commit()
    cur.close()
    conn.close()

def import_sql_text(
    host: str,
    port: int,
    admin_user: str,
    admin_pass: str,
    db_name: str,
    sql_text: str,
) -> Tuple[int, int]:
    """
    Import SQL into db_name, statement by statement.
    Returns (ok_count, fail_count).
    """
    if has_delimiter(sql_text):
        raise RuntimeError(
            "This .sql uses DELIMITER (procedures/triggers). "
            "For now, import those via MySQL Shell/CLI. "
            "Use a plain .sql dump without DELIMITER, or remove procedures."
        )

    statements = split_sql_statements(sql_text)
    if not statements:
        return (0, 0)

    conn = mysql_admin_connect(host, port, admin_user, admin_pass, database=db_name)
    cur = conn.cursor()
    ok = 0
    fail = 0

    try:
        for stmt in statements:
            st_clean = stmt.strip()
            if not st_clean:
                continue

            # Some dumps include USE db; ignore since we already select DB
            if re.match(r"(?i)^\s*use\s+", st_clean):
                continue

            cur.execute(st_clean)
            ok += 1

        conn.commit()
    except MySQLError as e:
        conn.rollback()
        fail += 1
        raise e
    finally:
        cur.close()
        conn.close()

    return ok, fail

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — Upload DB + Query + DataFrame")

# ---- Sidebar inputs (supports Railway variable names you showed)
with st.sidebar:
    st.header("OpenAI")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=safe_env("OPENAI_API_KEY", ""))
    model = st.text_input("OPENAI_MODEL", value=safe_env("OPENAI_MODEL", "gpt-4o-mini"))

    st.divider()
    st.header("MySQL Admin (root) Connection")

    mysql_host = st.text_input("MYSQL_HOST", value=safe_env("MYSQL_HOST", "127.0.0.1"))
    mysql_port_raw = st.text_input("MYSQL_PORT", value=safe_env("MYSQL_PORT", "3306"))
    mysql_port = parse_port(mysql_port_raw, default=3306)
    st.caption(f"Using port: {mysql_port}")

    mysql_root_user = st.text_input("MYSQL_ROOT_USER", value=safe_env("MYSQL_ROOT_USER", "root"))
    mysql_root_pass = st.text_input("MYSQL_ROOT_PASSWORD", type="password", value=safe_env("MYSQL_ROOT_PASSWORD", ""))

    st.divider()
    st.header("NLQ (read-only) User")
    nlq_user = st.text_input("NLQ_USER", value=safe_env("NLQ_USER", "nlq_user"))
    nlq_pass = st.text_input("NLQ_PASSWORD", type="password", value=safe_env("NLQ_PASSWORD", ""))

    st.divider()
    st.header("Database")
    db_name = st.text_input("MYSQL_DEFAULT_DB", value=safe_env("MYSQL_DEFAULT_DB", "Chinook"))

    st.divider()
    st.subheader("Upload & Import .sql")
    sql_upload = st.file_uploader("Upload a .sql file", type=["sql"])
    import_btn = st.button("Import SQL File (Create DB → Import → Create NLQ User → Grant SELECT)", type="primary")

    st.divider()
    test_admin_btn = st.button("Test Admin (root) Connection")
    test_nlq_btn = st.button("Test NLQ (read-only) Connection")

# ---- Connection tests
def test_mysql_connector(host: str, port: int, user: str, password: str, database: Optional[str] = None) -> None:
    conn = mysql_admin_connect(host, port, user, password, database=database)
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    cur.fetchone()
    cur.close()
    conn.close()

def get_engine(user: str, password: str, host: str, port: int, db: str) -> Engine:
    uri = build_mysql_uri(user, password, host, port, db)
    return create_engine(uri, pool_pre_ping=True)

if test_admin_btn:
    try:
        if not mysql_host.strip():
            raise RuntimeError("MYSQL_HOST is empty.")
        test_mysql_connector(mysql_host, mysql_port, mysql_root_user, mysql_root_pass, database=None)
        st.sidebar.success("Admin connection OK ✅")
    except Exception as e:
        st.sidebar.error(f"Admin connection failed: {e}")

if test_nlq_btn:
    try:
        if not mysql_host.strip():
            raise RuntimeError("MYSQL_HOST is empty.")
        test_mysql_connector(mysql_host, mysql_port, nlq_user, nlq_pass, database=db_name)
        st.sidebar.success("NLQ connection OK ✅")
    except Exception as e:
        st.sidebar.error(f"NLQ connection failed: {e}")

# ---- Import flow (one click)
if import_btn:
    if not mysql_host.strip():
        st.sidebar.error("MYSQL_HOST is empty.")
    elif not str(mysql_port).strip():
        st.sidebar.error("MYSQL_PORT is empty.")
    elif not mysql_root_user.strip() or not mysql_root_pass.strip():
        st.sidebar.error("Provide MYSQL_ROOT_USER and MYSQL_ROOT_PASSWORD.")
    elif not nlq_user.strip() or not nlq_pass.strip():
        st.sidebar.error("Provide NLQ_USER and NLQ_PASSWORD.")
    elif not db_name.strip():
        st.sidebar.error("Provide MYSQL_DEFAULT_DB (database name).")
    elif not sql_upload:
        st.sidebar.error("Upload a .sql file first.")
    else:
        try:
            sql_text = sql_upload.getvalue().decode("utf-8", errors="replace")

            with st.sidebar:
                with st.spinner("1/4 Creating database (if needed)..."):
                    create_database_if_needed(mysql_host, mysql_port, mysql_root_user, mysql_root_pass, db_name)

                with st.spinner("2/4 Importing SQL into database..."):
                    ok, _ = import_sql_text(mysql_host, mysql_port, mysql_root_user, mysql_root_pass, db_name, sql_text)

                with st.spinner("3/4 Creating NLQ user + granting SELECT..."):
                    ensure_nlq_user_and_grants(
                        mysql_host, mysql_port, mysql_root_user, mysql_root_pass, nlq_user, nlq_pass, db_name
                    )

                with st.spinner("4/4 Testing NLQ user connection..."):
                    test_mysql_connector(mysql_host, mysql_port, nlq_user, nlq_pass, database=db_name)

                st.success(f"Import complete ✅  Statements executed: {ok}.  NLQ user ready on `{db_name}`.")
        except Exception as e:
            st.sidebar.error(f"Import failed ❌\n{e}")
            st.sidebar.info(
                "Tip: If your SQL has `DELIMITER` / procedures / triggers, this app blocks it.\n"
                "Use MySQL Shell/CLI to import that kind of script, or export a plain schema+data dump."
            )

# ============================================================
# Main: Ask question -> Generate SQL -> Run -> DataFrame
# ============================================================

st.subheader("Ask a question")
question = st.text_area(
    "Example: Top 10 most rented films / Which country's customers spent the most by invoice?",
    height=90,
)
run_btn = st.button("Run Query")

if run_btn:
    if not api_key.strip():
        st.error("Please enter OPENAI_API_KEY in the sidebar.")
        st.stop()

    if not mysql_host.strip():
        st.error("MYSQL_HOST is empty.")
        st.stop()

    if not db_name.strip():
        st.error("MYSQL_DEFAULT_DB (database name) is empty.")
        st.stop()

    # Use NLQ user for querying (safer)
    try:
        engine = get_engine(nlq_user, nlq_pass, mysql_host, mysql_port, db_name)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        st.error(
            "NLQ user connection failed. Make sure you imported DB and granted privileges.\n\n"
            f"Error: {e}"
        )
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
