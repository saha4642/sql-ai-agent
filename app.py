import io
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_openai import ChatOpenAI

# ============================================================
# Streamlit config MUST be first Streamlit command
# ============================================================
st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")


# ============================================================
# 0) Helpers: env + engines (RO for query, RW for imports)
# ============================================================

def env_first(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def build_mysql_url(user: str, pw: str, host: str, port: str) -> str:
    u = quote_plus(user)
    p = quote_plus(pw)
    # no database in URL; we USE <db> per query
    return f"mysql+mysqlconnector://{u}:{p}@{host}:{port}/"


def require_base_mysql() -> Tuple[str, str]:
    host = env_first("MYSQL_HOST", "MYSQLHOST")
    port = env_first("MYSQL_PORT", "MYSQLPORT")
    if not host or not port:
        raise ValueError("Missing MYSQL_HOST/MYSQL_PORT in Railway variables.")
    return host, port


@st.cache_resource(show_spinner=False)
def get_engines() -> Dict[str, object]:
    """
    engine_ro: used for schema + NL queries (APP_MYSQL_*)
    engine_rw: used for imports (IMPORT_MYSQL_*) if available, else None
    """
    host, port = require_base_mysql()

    ro_user = env_first("APP_MYSQL_USER", "MYSQL_USER", "MYSQLUSER")
    ro_pw = env_first("APP_MYSQL_PASSWORD", "MYSQL_PASSWORD", "MYSQLPASSWORD")
    if not ro_user or not ro_pw:
        raise ValueError("Missing APP_MYSQL_USER/APP_MYSQL_PASSWORD (read-only creds).")

    ro_url = build_mysql_url(ro_user, ro_pw, host, port)
    engine_ro = create_engine(ro_url, pool_pre_ping=True, pool_recycle=1800)

    rw_user = env_first("IMPORT_MYSQL_USER")
    rw_pw = env_first("IMPORT_MYSQL_PASSWORD")
    engine_rw = None
    if rw_user and rw_pw:
        rw_url = build_mysql_url(rw_user, rw_pw, host, port)
        engine_rw = create_engine(rw_url, pool_pre_ping=True, pool_recycle=1800)

    return {
        "engine_ro": engine_ro,
        "engine_rw": engine_rw,
        "host": host,
        "port": port,
        "ro_user": ro_user,
        "rw_user": rw_user or "",
    }


def test_engine(engine: Engine) -> Tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ============================================================
# 1) DB discovery + schema
# ============================================================

SYSTEM_DBS = {"information_schema", "performance_schema", "mysql", "sys"}


def list_databases(engine: Engine) -> List[str]:
    df = pd.read_sql(text("SHOW DATABASES"), engine)
    dbs = sorted(df.iloc[:, 0].astype(str).tolist())
    return [d for d in dbs if d not in SYSTEM_DBS]


@dataclass(frozen=True)
class DBSchema:
    db: str
    tables: Dict[str, Set[str]]


def load_schema(engine: Engine, db: str) -> DBSchema:
    q = text("""
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    df = pd.read_sql(q, engine, params={"db": db})
    tables: Dict[str, Set[str]] = {}
    for t, g in df.groupby("TABLE_NAME"):
        tables[t] = set(g["COLUMN_NAME"].astype(str).tolist())
    return DBSchema(db=db, tables=tables)


def schema_to_text(schema: DBSchema, max_tables: int = 200) -> str:
    if not schema.tables:
        return "(No tables found.)"
    lines = []
    for i, (t, cols) in enumerate(schema.tables.items()):
        if i >= max_tables:
            lines.append("... (schema truncated)")
            break
        lines.append(f"- {t}: {', '.join(sorted(cols))}")
    return "\n".join(lines)


# ============================================================
# 2) Importers: CSV + URL
# ============================================================

IDENT_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def validate_identifier(name: str, label: str) -> str:
    s = (name or "").strip()
    if not IDENT_RE.match(s):
        raise ValueError(f"{label} must be alphanumeric/underscore (1–64 chars). Got: {name!r}")
    return s


def import_csv_df(engine_rw: Engine, db: str, table: str, df: pd.DataFrame, if_exists: str) -> int:
    with engine_rw.begin() as conn:
        conn.execute(text(f"USE `{db}`"))
        df.to_sql(table, con=conn, if_exists=if_exists, index=False, method="multi", chunksize=1000)
    return len(df)


SQL_DANGEROUS_RE = re.compile(
    r"\b(drop\s+database|drop\s+user|create\s+user|grant|revoke|flush\s+privileges)\b",
    re.IGNORECASE,
)


def split_sql_statements(sql_text: str) -> List[str]:
    parts = [p.strip() for p in (sql_text or "").split(";")]
    return [p + ";" for p in parts if p]


def download_url(url: str, max_mb: int = 30) -> Tuple[bytes, str]:
    url = (url or "").strip()
    if not url:
        raise ValueError("URL is empty.")

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    content = r.content
    if len(content) > max_mb * 1024 * 1024:
        raise ValueError(f"File too large (> {max_mb} MB).")

    path = urlparse(url).path.lower()
    if path.endswith(".csv"):
        kind = "csv"
    elif path.endswith(".sql"):
        kind = "sql"
    else:
        txt = content[:2000].decode("utf-8", errors="ignore").lower()
        kind = "sql" if "create table" in txt or "insert into" in txt else "csv"
    return content, kind


def import_sql_text(engine_rw: Engine, db: str, sql_text: str) -> Tuple[int, List[str]]:
    if SQL_DANGEROUS_RE.search(sql_text or ""):
        raise ValueError(
            "Blocked SQL import: contains dangerous statements (GRANT/CREATE USER/DROP DATABASE/etc). "
            "Use CLI for those."
        )

    stmts = split_sql_statements(sql_text)
    errors: List[str] = []
    ok_count = 0

    with engine_rw.begin() as conn:
        conn.execute(text(f"USE `{db}`"))
        for s in stmts:
            low = s.strip().lower()
            if not low or low.startswith("--") or low.startswith("/*"):
                continue
            try:
                conn.execute(text(s))
                ok_count += 1
            except Exception as e:
                errors.append(f"{str(e)}\n---\n{s[:2000]}")
    return ok_count, errors


# ============================================================
# 3) NL → SQL agent
# ============================================================

SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out or "")
    return m.group(1).strip() if m else (text_out or "").strip()


def normalize_sql(sql: str) -> str:
    return (sql or "").strip().rstrip(";").strip() + ";"


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())
    if re.search(r"\blimit\s+\d+\b", low):
        return s
    if low.startswith("select") and "count(" not in low:
        return s.rstrip(";") + f" LIMIT {default_limit};"
    return s


def is_read_only_sql(sql: str) -> bool:
    low = re.sub(r"\s+", " ", (sql or "").strip().lower())
    if not (low.startswith("select") or low.startswith("with")):
        return False
    blocked = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
        "grant", "revoke", "commit", "rollback", "call", "load data", "outfile"
    ]
    return not any(re.search(rf"\b{re.escape(k)}\b", low) for k in blocked)


def get_llm(api_key: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.0)


def generate_sql_from_question(llm: ChatOpenAI, schema: DBSchema, question: str) -> str:
    system = (
        "You are a senior data analyst writing MySQL queries.\n"
        "Rules:\n"
        "- Return ONLY the SQL query (no explanation).\n"
        "- SQL MUST be read-only (SELECT or WITH only).\n"
        "- Use only tables/columns from the provided schema.\n"
        "- Do not invent tables.\n"
    )
    user = f"TARGET DATABASE: {schema.db}\n\nSCHEMA:\n{schema_to_text(schema)}\n\nQUESTION:\n{question}\n\nReturn ONLY SQL."
    raw = llm.invoke(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}]
    ).content
    return ensure_limit(extract_sql(raw), default_limit=200)


def run_sql_to_df(engine_ro: Engine, db: str, sql: str) -> pd.DataFrame:
    with engine_ro.connect() as conn:
        conn.execute(text(f"USE `{db}`"))
        return pd.read_sql(text(sql), conn)


# ============================================================
# 4) UI
# ============================================================

st.title("Natural Language → SQL Agent (MySQL) — Upload → Import → Query")

eng = get_engines()
engine_ro: Engine = eng["engine_ro"]
engine_rw: Optional[Engine] = eng["engine_rw"]

with st.sidebar:
    st.header("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    st.divider()
    st.header("MySQL")
    st.write("Host:", eng["host"])
    st.write("Port:", eng["port"])
    st.write("RO user:", eng["ro_user"])
    st.write("RW user:", eng["rw_user"] if eng["rw_user"] else "(not set)")
    if st.button("Clear Cache"):
        get_engines.clear()
        st.success("Cache cleared ✅")
        st.stop()

with st.expander("Connection status", expanded=True):
    ok_ro, msg_ro = test_engine(engine_ro)
    st.write(f"Read-only engine: {'✅ OK' if ok_ro else '❌ FAIL'}")
    if not ok_ro:
        st.code(msg_ro)
        st.stop()

    if engine_rw is None:
        st.warning("Import engine not configured. Set IMPORT_MYSQL_USER / IMPORT_MYSQL_PASSWORD to enable uploads.")
    else:
        ok_rw, msg_rw = test_engine(engine_rw)
        st.write(f"Import engine: {'✅ OK' if ok_rw else '❌ FAIL'}")
        if not ok_rw:
            st.code(msg_rw)

dbs = list_databases(engine_ro)
if not dbs:
    st.error("No accessible databases found.")
    st.stop()

default_db = "sakila" if "sakila" in dbs else dbs[0]
selected_db = st.selectbox("Select database", dbs, index=dbs.index(default_db))

schema = load_schema(engine_ro, selected_db)
with st.expander("Schema preview", expanded=False):
    st.code(schema_to_text(schema), language="text")

st.subheader("Import data into MySQL")

tab_upload, tab_url = st.tabs(["Upload CSV (Desktop)", "Import from URL (.csv or simple .sql)"])

with tab_upload:
    table_name = st.text_input("Target table name", value="uploaded_data")
    if_exists = st.selectbox("If table exists", ["replace", "append"], index=0)
    file = st.file_uploader("Choose a CSV file", type=["csv"])

    if st.button("Import CSV → MySQL", disabled=(file is None or engine_rw is None)):
        try:
            t = validate_identifier(table_name, "Table name")
            df = pd.read_csv(file)
            rows = import_csv_df(engine_rw, selected_db, t, df, if_exists=if_exists)
            st.success(f"Imported {rows} rows into `{selected_db}.{t}` ✅")
        except Exception as e:
            st.error(f"CSV import failed: {e}")

with tab_url:
    url = st.text_input("URL to .csv or .sql")
    table_name_url = st.text_input("Table name (for CSV imports only)", value="url_data")
    if_exists_url = st.selectbox("If table exists (CSV)", ["replace", "append"], index=0)

    if st.button("Download & Import", disabled=(not url or engine_rw is None)):
        try:
            content, kind = download_url(url)
            if kind == "csv":
                t = validate_identifier(table_name_url, "Table name")
                df = pd.read_csv(io.BytesIO(content))
                rows = import_csv_df(engine_rw, selected_db, t, df, if_exists=if_exists_url)
                st.success(f"Imported {rows} rows into `{selected_db}.{t}` ✅")
            else:
                sql_text = content.decode("utf-8", errors="ignore")
                ok_count, errors = import_sql_text(engine_rw, selected_db, sql_text)
                st.success(f"Executed {ok_count} SQL statements into `{selected_db}` ✅")
                if errors:
                    st.warning(f"{len(errors)} statements failed. First error:")
                    st.code(errors[0])
        except Exception as e:
            st.error(f"URL import failed: {e}")

st.subheader("Ask a question (NL → SQL)")
question = st.text_area("Example (Sakila): Which customers paid the most in total?", height=90)

if st.button("Run NL → SQL Query"):
    if not openai_key.strip():
        st.error("Missing OPENAI_API_KEY.")
        st.stop()

    schema = load_schema(engine_ro, selected_db)
    if not schema.tables:
        st.error(f"No tables found in `{selected_db}`. Import data first.")
        st.stop()

    llm = get_llm(openai_key, openai_model)

    with st.spinner("Generating SQL..."):
        sql = generate_sql_from_question(llm, schema, question)

    st.markdown("### Generated SQL")
    st.code(sql, language="sql")

    if not is_read_only_sql(sql):
        st.error("Blocked: generated SQL is not read-only (SELECT/WITH only).")
        st.stop()

    with st.spinner("Running query..."):
        try:
            df = run_sql_to_df(engine_ro, selected_db, sql)
        except Exception as e:
            st.error(f"SQL execution failed: {e}")
            st.stop()

    st.markdown("### Results")
    st.dataframe(df, use_container_width=True)
