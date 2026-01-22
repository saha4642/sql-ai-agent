from __future__ import annotations

import os
import re
import socket
from typing import Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine import URL


# ============================================================
# 0) DB config (Railway, non-root)
#    - Uses discrete vars: MYSQL_HOST/PORT/DATABASE
#    - Uses APP_MYSQL_USER/APP_MYSQL_PASSWORD as credentials
#    - Forces mysql-connector (supports caching_sha2_password; needs cryptography)
# ============================================================

def env_any(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def load_db_cfg() -> dict:
    user = env_any("APP_MYSQL_USER")
    pw = env_any("APP_MYSQL_PASSWORD")

    host = env_any("MYSQL_HOST")
    port = env_any("MYSQL_PORT")
    db = env_any("MYSQL_DATABASE")

    return {"user": user, "pw": pw, "host": host, "port": port, "db": db}


def build_sqlalchemy_url(cfg: dict) -> URL:
    missing = [k for k in ("user", "pw", "host", "port", "db") if not (cfg.get(k) or "").strip()]
    if missing:
        raise RuntimeError(f"Missing required DB vars: {', '.join(missing)}")

    if cfg["user"].lower() == "root":
        raise RuntimeError(
            "APP_MYSQL_USER is 'root'. Railway blocks root for app connections. "
            "Use the non-root user you created (app_ro)."
        )

    # URL.create avoids all quoting/encoding issues
    return URL.create(
        drivername="mysql+mysqlconnector",
        username=cfg["user"],
        password=cfg["pw"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["db"],
    )


@st.cache_resource(show_spinner=False)
def get_engine(url: URL) -> Engine:
    # mysql-connector supports caching_sha2_password, but requires `cryptography`
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
    return (sql or "").strip().rstrip(";") + ";"


def is_read_only_sql(sql: str) -> bool:
    low = re.sub(r"\s+", " ", (sql or "").strip().lower())
    if not (low.startswith("select") or low.startswith("with")):
        return False
    blocked = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
        "grant", "revoke", "commit", "rollback", "call", "load data", "outfile",
    ]
    return not any(re.search(rf"\b{b}\b", low) for b in blocked)


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    s = normalize_sql(sql)
    low = re.sub(r"\s+", " ", s.lower())
    if re.search(r"\blimit\s+\d+\b", low):
        return s
    if low.startswith("select") and "count(" not in low:
        return s.rstrip(";") + f" LIMIT {default_limit};"
    return s


def get_llm(api_key: str, model: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.0)


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
    sql = ensure_limit(sql, default_limit=200)
    return sql


def run_sql_to_df(engine: Engine, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)


# ============================================================
# 2) Streamlit UI
# ============================================================

st.set_page_config(page_title="NL → SQL (MySQL) Agent", layout="wide")
st.title("Natural Language → SQL Agent (MySQL) — DataFrame Results")

with st.sidebar:
    st.header("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=env_any("OPENAI_API_KEY"))
    openai_model = st.text_input("OPENAI_MODEL", value=env_any("OPENAI_MODEL") or "gpt-4o-mini")

    st.divider()
    st.header("MySQL (Railway)")
    clear_cache = st.button("Clear DB Cache")
    test_btn = st.button("Test Connection")

    st.divider()
    st.header("DB Debug (safe)")
    st.write("APP_MYSQL_USER set:", bool(env_any("APP_MYSQL_USER")))
    st.write("APP_MYSQL_PASSWORD set:", bool(env_any("APP_MYSQL_PASSWORD")))
    st.write("MYSQL_HOST present:", bool(env_any("MYSQL_HOST")))
    st.write("MYSQL_PORT present:", bool(env_any("MYSQL_PORT")))
    st.write("MYSQL_DATABASE present:", bool(env_any("MYSQL_DATABASE")))

if clear_cache:
    get_engine.clear()
    st.success("DB cache cleared ✅")

cfg = load_db_cfg()

# ============================================================
# 3) Connection status (with DNS debug)
# ============================================================

with st.expander("Connection status", expanded=True):
    st.write("Using creds: APP_MYSQL_*")
    st.write("Using user:", cfg["user"] or "(missing)")
    st.write("Host:", cfg["host"] or "(missing)", "Port:", cfg["port"] or "(missing)", "DB:", cfg["db"] or "(missing)")

    # DNS resolution helps confirm app points to expected internal endpoint
    try:
        addrs = socket.getaddrinfo(cfg["host"], int(cfg["port"]), proto=socket.IPPROTO_TCP)
        uniq = sorted({a[4][0] for a in addrs})
        st.write("DNS resolves to:", uniq[:10])
    except Exception as e:
        st.write("DNS resolution failed:", str(e))

    try:
        url = build_sqlalchemy_url(cfg)
        engine = get_engine(url)
        ok, msg = test_engine(engine)
        st.write(f"Engine: {'✅ OK' if ok else '❌ FAIL'}")
        if not ok:
            st.code(msg)
    except Exception as e:
        st.write("Engine: ❌ FAIL")
        st.code(str(e))
        st.stop()

if test_btn:
    # just forces the block above to run after a click
    st.toast("Test completed (see Connection status).", icon="✅")

# ============================================================
# 4) Main: NL -> SQL -> DataFrame
# ============================================================

st.subheader("Ask a question")
question = st.text_area(
    "Example: Which country's customers spent the most by invoice?",
    height=90,
)

if st.button("Run Query"):
    if not openai_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()

    llm = get_llm(openai_key, openai_model)

    with st.spinner("Reading schema..."):
        schema = fetch_schema_summary(engine, db=cfg["db"])

    with st.spinner("Generating SQL..."):
        sql = generate_sql_from_question(llm, schema, question)

    st.markdown("### Generated SQL")
    st.code(sql, language="sql")

    if not is_read_only_sql(sql):
        st.error("Blocked: generated SQL is not read-only (SELECT/WITH only).")
        st.stop()

    with st.spinner("Running query..."):
        df = run_sql_to_df(engine, sql)

    st.markdown("### Results")
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
