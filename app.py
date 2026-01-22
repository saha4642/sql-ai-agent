from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import Error as MySQLError

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url, URL

from langchain_openai import ChatOpenAI

# ----------------------------
# Regex
# ----------------------------
SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# ----------------------------
# Models
# ----------------------------
@dataclass
class DbConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


# ----------------------------
# Env helpers
# ----------------------------
def _get_env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


# ----------------------------
# URL normalization (Railway-safe)
# ----------------------------
def normalize_mysql_sqlalchemy_url(url_str: str) -> str:
    """
    Railway often provides:
      mysql://user:pass@host:port/db

    SQLAlchemy should use a driver:
      mysql+mysqlconnector://user:pass@host:port/db
    """
    if not url_str or not isinstance(url_str, str):
        raise ValueError("Empty database URL")

    s = url_str.strip()

    # If user pasted an unresolved Railway template rather than a real reference
    if s.startswith("${{") or s.startswith("{{"):
        raise ValueError(
            "DATABASE_URL is an unresolved Railway template string. "
            "In Railway -> Variables, set it using 'Add Reference' to MySQL.MYSQL_URL "
            "instead of typing ${{ MySQL.MYSQL_URL }}."
        )

    # Convert mysql:// -> mysql+mysqlconnector://
    if s.startswith("mysql://"):
        s = s.replace("mysql://", "mysql+mysqlconnector://", 1)

    return s


def parse_sqlalchemy_url(url_str: str) -> Tuple[str, DbConfig]:
    normalized = normalize_mysql_sqlalchemy_url(url_str)
    u = make_url(normalized)
    cfg = DbConfig(
        host=u.host or "127.0.0.1",
        port=int(u.port or 3306),
        user=u.username or "",
        password=u.password or "",
        database=(u.database or "").lstrip("/"),
    )
    return normalized, cfg


def set_db_on_url(url_str: str, db_name: str) -> str:
    """
    Ensure the URL points to db_name.
    """
    normalized = normalize_mysql_sqlalchemy_url(url_str)
    u = make_url(normalized)
    u2 = u.set(database=db_name)
    return str(u2)


def build_mysqlconnector_url(user: str, password: str, host: str, port: int, db: str) -> str:
    return str(
        URL.create(
            "mysql+mysqlconnector",
            username=user,
            password=password,
            host=host,
            port=int(port),
            database=db,
        )
    )


def make_engine(url_str: str) -> Engine:
    return create_engine(url_str, pool_pre_ping=True)


# ----------------------------
# LLM
# ----------------------------
def get_llm(api_key: str, model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)


# ----------------------------
# SQL generation helpers
# ----------------------------
def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out)
    if m:
        return m.group(1).strip()
    return text_out.strip()


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

    # Add a default LIMIT for non-count SELECTs
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

    lines: List[str] = []
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


# ----------------------------
# Import helpers
# ----------------------------
def mysql_root_connect(root_cfg: DbConfig, database: Optional[str] = None):
    kwargs = dict(
        host=str(root_cfg.host).strip(),
        port=int(root_cfg.port),
        user=str(root_cfg.user).strip(),
        password=root_cfg.password,
    )
    if database:
        kwargs["database"] = database
    return mysql.connector.connect(**kwargs)


def create_database(root_cfg: DbConfig, db_name: str) -> None:
    conn = mysql_root_connect(root_cfg)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
    conn.commit()
    cur.close()
    conn.close()


def import_sql_text(root_cfg: DbConfig, db_name: str, sql_text: str) -> int:
    """
    Imports SQL using mysql-connector's multi=True.
    Returns number of statements iterated.
    """
    conn = mysql_root_connect(root_cfg, database=db_name)
    conn.autocommit = False
    cur = conn.cursor()
    executed = 0
    try:
        for _ in cur.execute(sql_text, multi=True):
            executed += 1
        conn.commit()
        return executed
    except MySQLError:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def ensure_nlq_user_and_grants(root_cfg: DbConfig, db_name: str, nlq_user: str, nlq_password: str) -> None:
    """
    Creates nlq_user@'%' and grants SELECT, SHOW VIEW on db.
    This is required on Railway because client host is not localhost.
    """
    conn = mysql_root_connect(root_cfg, database=db_name)
    cur = conn.cursor()

    # Ensure user exists for any host
    cur.execute(f"CREATE USER IF NOT EXISTS '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_password,))
    # Ensure password is updated if user already existed
    cur.execute(f"ALTER USER '{nlq_user}'@'%' IDENTIFIED BY %s;", (nlq_password,))

    # Grants
    cur.execute(f"GRANT SELECT, SHOW VIEW ON `{db_name}`.* TO '{nlq_user}'@'%';")
    # Some installations benefit from this for schema queries
    cur.execute(f"GRANT SELECT ON `information_schema`.* TO '{nlq_user}'@'%';")

    cur.execute("FLUSH PRIVILEGES;")
    conn.commit()

    cur.close()
    conn.close()


def read_uploaded_sql_files(upload) -> List[Tuple[str, str]]:
    """
    Accepts:
      - .sql => [(filename, sql_text)]
      - .zip => list of .sql inside, sorted so schema loads before data if possible
    """
    name = upload.name.lower()
    raw = upload.getvalue()

    if name.endswith(".sql"):
        return [(upload.name, raw.decode("utf-8", errors="replace"))]

    if name.endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(raw))
        sql_members = [m for m in zf.namelist() if m.lower().endswith(".sql")]
        if not sql_members:
            return []

        def sort_key(p: str) -> Tuple[int, str]:
            pl = p.lower()
            if "schema" in pl:
                return (0, pl)
            if "data" in pl:
                return (1, pl)
            return (2, pl)

        out: List[Tuple[str, str]] = []
        for m in sorted(sql_members, key=sort_key):
            out.append((m, zf.read(m).decode("utf-8", errors="replace")))
        return out

    return []


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — Upload DB + Query + DataFrame")

with st.sidebar:
    st.header("OpenAI")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=_get_env("OPENAI_API_KEY"))
    model = st.text_input("OPENAI_MODEL", value=_get_env("OPENAI_MODEL", default="gpt-4o-mini"))

    st.divider()
    st.header("MySQL (Railway)")
    st.caption("Set DATABASE_URL_ROOT using Railway variable reference to MySQL.MYSQL_URL.")
    show_debug = st.checkbox("Show resolved DB host/port (safe)", value=False)

    st.divider()
    st.header("Target database")
    db_name = st.text_input("DB name", value=_get_env("MYSQL_DEFAULT_DB", "MYSQLDATABASE", "MYSQL_DATABASE", default="sakila"))

    st.divider()
    st.header("NLQ read-only user")
    nlq_user = st.text_input("NLQ_USER", value=_get_env("NLQ_USER", default="nlq_user"))
    nlq_password = st.text_input("NLQ_PASSWORD", type="password", value=_get_env("NLQ_PASSWORD", default=""))

    st.divider()
    st.subheader("Upload & Import")
    sql_upload = st.file_uploader("Upload a .sql file or .zip of .sql files", type=["sql", "zip"])
    do_import = st.button("Import SQL File (Create DB → Import → Create NLQ User → Grant SELECT)", use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    do_test_admin = c1.button("Test Admin", use_container_width=True)
    do_test_nlq = c2.button("Test NLQ", use_container_width=True)

# ----------------------------
# Resolve root/admin URL from env
# ----------------------------
root_url_raw = _get_env("DATABASE_URL_ROOT", "DATABASE_URL", "MYSQL_URL", "MYSQL_PUBLIC_URL", default="")

if not root_url_raw:
    st.error(
        "Missing DATABASE_URL_ROOT.\n\n"
        "In Railway -> sql-ai-agent -> Variables:\n"
        "- Add DATABASE_URL_ROOT as a Reference to MySQL.MYSQL_URL"
    )
    st.stop()

try:
    root_url_norm, root_cfg = parse_sqlalchemy_url(root_url_raw)
except Exception as e:
    st.error(f"Invalid DATABASE_URL_ROOT: {e}")
    st.stop()

if show_debug:
    st.sidebar.write(
        {
            "root_host": root_cfg.host,
            "root_port": root_cfg.port,
            "root_user": root_cfg.user,
            "root_db_in_url": root_cfg.database,
        }
    )

def test_engine(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

def root_engine_for_db(db: str) -> Engine:
    url = set_db_on_url(root_url_norm, db)
    return make_engine(url)

def nlq_engine_for_db(db: str) -> Engine:
    if not nlq_password.strip():
        raise ValueError("NLQ_PASSWORD is empty")
    url = build_mysqlconnector_url(nlq_user, nlq_password, root_cfg.host, int(root_cfg.port), db)
    return make_engine(url)

# ----------------------------
# Sidebar actions
# ----------------------------
if do_test_admin:
    try:
        e = root_engine_for_db(db_name or (root_cfg.database or "mysql"))
        test_engine(e)
        st.sidebar.success("Admin connection OK ✅")
    except Exception as e:
        st.sidebar.error(f"Admin connection failed: {e}")

if do_test_nlq:
    if not nlq_password.strip():
        st.sidebar.warning("Set NLQ_PASSWORD first (Railway variable).")
    else:
        try:
            e = nlq_engine_for_db(db_name.strip())
            test_engine(e)
            st.sidebar.success("NLQ connection OK ✅")
        except Exception as e:
            st.sidebar.error(f"NLQ connection failed: {e}")

if do_import:
    if not sql_upload:
        st.sidebar.warning("Upload a .sql or .zip first.")
    elif not nlq_password.strip():
        st.sidebar.warning("Set NLQ_PASSWORD first (Railway variable).")
    elif not db_name.strip():
        st.sidebar.warning("Set a DB name.")
    else:
        try:
            # 1) create DB
            create_database(root_cfg, db_name)

            # 2) import SQL files
            files = read_uploaded_sql_files(sql_upload)
            if not files:
                st.sidebar.error("No .sql files found in the upload.")
            else:
                total = 0
                for fname, sql_text in files:
                    with st.spinner(f"Importing {fname} ..."):
                        n = import_sql_text(root_cfg, db_name, sql_text)
                        total += n

                # 3) create nlq user + grants (Railway-safe: '%' host)
                ensure_nlq_user_and_grants(root_cfg, db_name, nlq_user, nlq_password)

                st.sidebar.success(f"Import finished ✅  Files: {len(files)}  Statements executed: {total}")
                st.sidebar.info("Next: click Test NLQ, then run a question.")
        except Exception as e:
            st.sidebar.error(f"Import failed ❌\n{e}")
            st.sidebar.info(
                "Tip: If your SQL uses DELIMITER / stored procedures, mysql-connector multi import may fail.\n"
                "Use a dump with tables + inserts only."
            )

# ----------------------------
# Main: Ask question -> SQL -> DataFrame
# ----------------------------
st.subheader("Ask a question")
question = st.text_area(
    "Example: Top 10 most rented films (Sakila) / Which country's customers spent the most (Chinook)",
    height=90,
)
run_btn = st.button("Run Query")

if run_btn:
    if not api_key.strip():
        st.error("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()

    if not db_name.strip():
        st.error("Please set a database name in the sidebar.")
        st.stop()

    if not nlq_password.strip():
        st.error("Please set NLQ_PASSWORD in the sidebar (Railway variable).")
        st.stop()

    # Ensure DB exists + NLQ grants (idempotent)
    try:
        create_database(root_cfg, db_name)
        ensure_nlq_user_and_grants(root_cfg, db_name, nlq_user, nlq_password)
    except Exception as e:
        st.error(f"Admin setup failed (create db / nlq grants): {e}")
        st.stop()

    # Engines
    try:
        admin_engine = root_engine_for_db(db_name)
        query_engine = nlq_engine_for_db(db_name)
        test_engine(admin_engine)
        test_engine(query_engine)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

    llm = get_llm(api_key=api_key, model=model, temperature=0.0)

    with st.spinner("Reading schema..."):
        schema = fetch_schema_summary(admin_engine, db_name)

    if not schema.strip():
        st.warning("Your database has no tables yet. Import a .sql first.")
        st.stop()

    with st.spinner("Generating SQL..."):
        sql = generate_sql_from_question(llm, schema, question)

    st.markdown("### Generated SQL")
    st.code(sql, language="sql")

    if not is_read_only_sql(sql):
        st.error("Blocked: generated SQL is not read-only (SELECT/WITH only).")
        st.stop()

    with st.spinner("Running query (NLQ user)..."):
        try:
            df = run_sql_to_df(query_engine, sql)
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
