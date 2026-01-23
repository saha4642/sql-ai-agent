# app.py

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_openai import ChatOpenAI

# ============================================================
# IMPORTANT: set_page_config must be the first Streamlit command
# ============================================================
st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")

# ============================================================
# 0) DB URL build (Railway-friendly, non-root)
# ============================================================

def env_first(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return ""


def build_mysql_sqlalchemy_url() -> Tuple[str, Dict[str, str]]:
    """
    Prefer APP_MYSQL_* creds (your read-only user),
    fall back to MYSQL_* if needed.

    Uses private network vars (MYSQL_HOST mysql.railway.internal, port 3306, db railway by default).
    """
    user = env_first("APP_MYSQL_USER", "MYSQL_USER", "MYSQLUSER")
    pw = env_first("APP_MYSQL_PASSWORD", "MYSQL_PASSWORD", "MYSQLPASSWORD")
    host = env_first("MYSQL_HOST", "MYSQLHOST")
    port = env_first("MYSQL_PORT", "MYSQLPORT")
    db = env_first("MYSQL_DATABASE", "MYSQL_DB", "MYSQL_DEFAULT_DB", "MYSQLDATABASE")

    debug = {
        "user": user,
        "host": host,
        "port": port,
        "db": db,
        "using_creds": "APP_MYSQL_*" if os.getenv("APP_MYSQL_USER") else "MYSQL_*",
    }

    if not user or not pw or not host or not port or not db:
        missing = [k for k, v in debug.items() if k in {"user", "host", "port", "db"} and not v]
        raise ValueError(f"Missing required DB env vars: {missing}. Check Railway variables.")

    u = quote_plus(user)
    p = quote_plus(pw)
    # Engine URL includes default DB (for connection), but we will USE selected_db before running SQL.
    url = f"mysql+mysqlconnector://{u}:{p}@{host}:{port}/{db}"
    return url, debug


@st.cache_resource(show_spinner=False)
def get_engine() -> Tuple[Engine, Dict[str, str]]:
    url, dbg = build_mysql_sqlalchemy_url()
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=1800)
    return engine, dbg


def test_engine(engine: Engine) -> Tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=30, show_spinner=False)
def list_databases(_engine: Engine) -> List[str]:
    df = pd.read_sql(text("SHOW DATABASES"), _engine)
    dbs = sorted(df.iloc[:, 0].astype(str).tolist())
    hide = {"information_schema", "performance_schema", "mysql", "sys"}
    return [d for d in dbs if d not in hide]



# ============================================================
# 1) Schema introspection + SQL validation
# ============================================================

@dataclass(frozen=True)
class DBSchema:
    db: str
    tables: Dict[str, Set[str]]  # table -> set(columns)


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


def schema_to_prompt(schema: DBSchema, max_tables: int = 200) -> str:
    lines = []
    for i, (t, cols) in enumerate(schema.tables.items()):
        if i >= max_tables:
            lines.append("... (schema truncated)")
            break
        lines.append(f"- {t}: {', '.join(sorted(cols))}")
    if not lines:
        return "(No tables found in this database.)"
    return "\n".join(lines)


SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_sql(text_out: str) -> str:
    m = SQL_CODEBLOCK_RE.search(text_out or "")
    return m.group(1).strip() if m else (text_out or "").strip()


def normalize_sql(sql: str) -> str:
    s = (sql or "").strip().rstrip(";").strip()
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


# Very lightweight SQL name extraction (works for typical SELECTs)
TABLE_TOKEN_RE = re.compile(r"\bfrom\s+([`\"\w\.]+)|\bjoin\s+([`\"\w\.]+)", re.IGNORECASE)


def strip_quotes(name: str) -> str:
    return (name or "").strip().strip("`").strip('"')


def referenced_tables(sql: str) -> Set[str]:
    s = sql or ""
    out: Set[str] = set()
    for m in TABLE_TOKEN_RE.finditer(s):
        for g in m.groups():
            if g:
                t = strip_quotes(g)
                # handle db.table
                if "." in t:
                    t = t.split(".")[-1]
                out.add(t)
    return out


def validate_tables_exist(schema: DBSchema, sql: str) -> Tuple[bool, str]:
    tabs = referenced_tables(sql)
    missing = [t for t in tabs if t not in schema.tables]
    if missing:
        return False, f"Unknown table(s): {', '.join(missing)}"
    if not tabs:
        return False, "Could not detect any tables in SQL."
    return True, "OK"


# ============================================================
# 2) LLM: generate SQL with regeneration on schema violations
# ============================================================

def get_llm(api_key: str, model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)


def generate_sql(
    llm: ChatOpenAI,
    schema: DBSchema,
    question: str,
    prev_error: Optional[str] = None,
) -> str:
    system = (
        "You are a senior data analyst writing MySQL queries.\n"
        "Rules:\n"
        "- Return ONLY the SQL query (no explanation).\n"
        "- SQL MUST be read-only (SELECT or WITH only).\n"
        "- Use ONLY tables/columns that appear in the provided schema.\n"
        "- If the database has no tables, return: SELECT 'no tables' AS message;\n"
        "- Prefer simple queries that will run.\n"
    )

    schema_txt = schema_to_prompt(schema)
    user = f"SCHEMA:\n{schema_txt}\n\nQUESTION:\n{question}\n\nReturn ONLY SQL."

    if prev_error:
        user += f"\n\nIMPORTANT: Your last query failed with: {prev_error}\nFix it using ONLY the schema."

    raw = llm.invoke(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}]
    ).content

    sql = extract_sql(raw)
    sql = normalize_sql(sql)
    sql = ensure_limit(sql, default_limit=200)
    return sql


def run_sql_to_df(engine: Engine, db: str, sql: str) -> pd.DataFrame:
    """
    Ensure we run inside the selected database.
    This is what makes the dropdown actually control the query target.
    """
    with engine.connect() as conn:
        conn.execute(text(f"USE `{db}`"))
        return pd.read_sql(text(sql), conn)


# ============================================================
# 3) Streamlit UI
# ============================================================

st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in environment variables.")
    st.stop()


    st.divider()
    st.header("MySQL (Railway)")
    st.caption("Uses MYSQL_HOST/PORT/DATABASE + APP_MYSQL_USER/APP_MYSQL_PASSWORD. Root URLs are not used.")

    btn_clear = st.button("Clear DB Cache")
    btn_refresh_dbs = st.button("🔄 Refresh DB list")
    btn_test = st.button("Test Connection")
    btn_show_schema = st.button("Show Schema")


if btn_clear:
    get_engine.clear()
    list_databases.clear()
    st.sidebar.success("DB cache cleared ✅")


# Connect
try:
    engine, dbg = get_engine()
except Exception as e:
    st.error(f"DB config error: {e}")
    st.stop()


# Refresh DB list if requested
if btn_refresh_dbs:
    list_databases.clear()
    st.rerun()


# Load DBs visible to this user
try:
    dbs = list_databases(engine)
except Exception as e:
    st.error(f"Failed to list databases: {e}")
    st.stop()

if not dbs:
    st.error("No databases visible to this user.")
    st.stop()


# Default selection: prefer env db, else sakila, else first
env_db = dbg["db"]
if env_db in dbs:
    default_db = env_db
elif "sakila" in dbs:
    default_db = "sakila"
else:
    default_db = dbs[0]

selected_db = st.sidebar.selectbox(
    "Select database",
    dbs,
    index=dbs.index(default_db),
)

# Status box
with st.expander("Connection status", expanded=True):
    st.write(f"Using creds: **{dbg['using_creds']}**")
    st.write(f"Using user: **{dbg['user']}**")
    st.write(f"Host: **{dbg['host']}**  Port: **{dbg['port']}**")
    st.write(f"Env default DB: **{dbg['db']}**")
    st.write(f"Selected DB (dropdown): **{selected_db}**")

    if btn_test:
        ok, msg = test_engine(engine)
        st.write(f"Engine: {'✅ OK' if ok else '❌ FAIL'}")
        if not ok:
            st.code(msg)


# Load schema for selected DB
try:
    schema = load_schema(engine, db=selected_db)
except Exception as e:
    st.error(f"Failed to read schema from `{selected_db}`: {e}")
    st.stop()

if btn_show_schema:
    st.subheader(f"Database schema (detected) — {selected_db}")
    if not schema.tables:
        st.info("No tables found in this database yet.")
    else:
        st.code(schema_to_prompt(schema), language="text")


# Main question UI
st.subheader("Ask a question")
question = st.text_area(
    "Example (Sakila): Which customers paid the most in total?",
    height=90,
)
run_btn = st.button("Run Query")

if run_btn:
    if not openai_key.strip():
        st.error("Missing OPENAI_API_KEY.")
        st.stop()

    if not schema.tables:
        st.warning(f"Database `{selected_db}` has no tables. Create/import tables first, then ask questions.")
        st.stop()

    llm = get_llm(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.0)

    # Try up to 3 attempts to avoid hallucinated tables
    last_err: Optional[str] = None
    sql: str = ""

    for attempt in range(1, 4):
        with st.spinner(f"Generating SQL (attempt {attempt}/3)..."):
            sql = generate_sql(llm, schema, question, prev_error=last_err)

        st.markdown("### Generated SQL")
        st.code(sql, language="sql")

        if not is_read_only_sql(sql):
            last_err = "Non read-only SQL (must start with SELECT/WITH)."
            continue

        ok_tables, msg_tables = validate_tables_exist(schema, sql)
        if not ok_tables:
            last_err = msg_tables
            st.warning(f"Regenerating: {msg_tables}")
            continue

        # Execute
        with st.spinner("Running query..."):
            try:
                df = run_sql_to_df(engine, selected_db, sql)
                st.markdown("### Results (DataFrame)")
                st.dataframe(df, use_container_width=True)

                # Summarize
                with st.spinner("Summarizing results..."):
                    preview = df.head(20).to_csv(index=False)
                    summary_prompt = (
                        "Summarize the result for a business user in 2-5 bullet points.\n"
                        "If there are totals/rankings, mention the top items.\n\n"
                        f"Database: {selected_db}\n\n"
                        f"Question: {question}\n\n"
                        f"SQL:\n{sql}\n\n"
                        f"CSV Preview (first rows):\n{preview}"
                    )
                    summary = llm.invoke(summary_prompt).content

                st.markdown("### Summary")
                st.write(summary)
                break

            except Exception as e:
                last_err = str(e)
                st.warning("Query failed; regenerating a safer query using only schema…")
                continue
    else:
        st.error("Failed to produce a working query after 3 attempts. Try a simpler question or inspect schema.")
        if last_err:
            st.code(last_err)
