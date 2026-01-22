from __future__ import annotations

import os
import re
import socket
from typing import Tuple
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url


# -----------------------------
# Env helpers
# -----------------------------
def env_any(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


# -----------------------------
# Build DB URL (APP_* wins)
# -----------------------------
def build_mysql_components() -> dict:
    app_user = env_any("APP_MYSQL_USER")
    app_pw = env_any("APP_MYSQL_PASSWORD")

    # We intentionally DO NOT use MYSQL_USER/MYSQL_PASSWORD anymore.
    # Only APP_* should be credentials.
    host = env_any("MYSQL_HOST")
    port = env_any("MYSQL_PORT")
    db = env_any("MYSQL_DATABASE")

    using = "APP_MYSQL_*" if (app_user and app_pw) else "MISSING_APP_CREDS"

    return {
        "using": using,
        "user": app_user,
        "pw": app_pw,
        "host": host,
        "port": port,
        "db": db,
    }


def normalize_sqlalchemy_url(user: str, pw: str, host: str, port: str, db: str) -> str:
    if not all([user, pw, host, port, db]):
        raise RuntimeError("Missing one or more required DB fields (user/pw/host/port/db).")

    if user.lower() == "root":
        raise RuntimeError("Root is blocked. Use a non-root user (app_ro).")

    # Build raw mysql URL then normalize to mysql+pymysql
    raw = f"mysql://{user}:{pw}@{host}:{port}/{db}"
    s = raw.replace("mysql://", "mysql+pymysql://", 1)

    u = make_url(s)
    u = u.set(
        username=quote_plus(u.username) if u.username else None,
        password=quote_plus(u.password) if u.password else None,
    )
    return str(u)


@st.cache_resource(show_spinner=False)
def get_engine(sqlalchemy_url: str) -> Engine:
    return create_engine(sqlalchemy_url, pool_pre_ping=True, pool_recycle=1800)


def test_engine(engine: Engine) -> Tuple[bool, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "OK"
    except Exception as e:
        return False, str(e)


# -----------------------------
# SQL safety helpers
# -----------------------------
SQL_CODEBLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_sql(txt: str) -> str:
    m = SQL_CODEBLOCK_RE.search(txt or "")
    return m.group(1).strip() if m else (txt or "").strip()


def normalize_sql(sql: str) -> str:
    return (sql or "").strip().rstrip(";") + ";"


def is_read_only_sql(sql: str) -> bool:
    low = re.sub(r"\s+", " ", (sql or "").strip().lower())
    if not (low.startswith("select") or low.startswith("with")):
        return False
    banned = [
        "insert", "update", "delete", "drop", "alter", "truncate", "create", "replace",
        "grant", "revoke", "commit", "rollback", "call", "load data", "outfile",
    ]
    return not any(re.search(rf"\b{b}\b", low) for b in banned)


def ensure_limit(sql: str, limit: int = 200) -> str:
    s = normalize_sql(sql)
    low = s.lower()
    if re.search(r"\blimit\s+\d+\b", low):
        return s
    if low.startswith("select") and "count(" not in low:
        return s.rstrip(";") + f" LIMIT {limit};"
    return s


def get_llm(api_key: str, model: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.0)


def fetch_schema(engine: Engine, db: str) -> str:
    q = text("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    df = pd.read_sql(q, engine, params={"db": db})
    lines = []
    for t, g in df.groupby("TABLE_NAME"):
        cols = ", ".join(f"{r.COLUMN_NAME}({r.DATA_TYPE})" for r in g.itertuples(index=False))
        lines.append(f"- {t}: {cols}")
    return "\n".join(lines)


def generate_sql(llm, schema: str, question: str) -> str:
    system = (
        "You are a senior data analyst writing MySQL queries.\n"
        "Rules:\n"
        "- Return ONLY the SQL query (no explanation).\n"
        "- SQL MUST be read-only (SELECT or WITH only).\n"
        "- Use only tables/columns from the provided schema.\n"
    )
    user = f"SCHEMA:\n{schema}\n\nQUESTION:\n{question}\n\nReturn ONLY SQL."
    raw = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content
    sql = extract_sql(raw)
    sql = ensure_limit(sql, 200)
    return sql


# ============================================================
# Streamlit UI
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

cfg = build_mysql_components()

# Connection status expander
with st.expander("Connection status", expanded=True):
    st.write("Using creds:", cfg["using"])
    st.write("Using user:", cfg["user"] or "(missing)")
    st.write("Host:", cfg["host"] or "(missing)", "Port:", cfg["port"] or "(missing)", "DB:", cfg["db"] or "(missing)")

    # DNS resolution (works even if DB auth fails)
    try:
        addrs = socket.getaddrinfo(cfg["host"], int(cfg["port"]), proto=socket.IPPROTO_TCP)
        uniq = sorted({a[4][0] for a in addrs})
        st.write("DNS resolves to:", uniq[:10])
    except Exception as e:
        st.write("DNS resolution failed:", str(e))

    # Build engine + test
    try:
        sqlalchemy_url = normalize_sqlalchemy_url(cfg["user"], cfg["pw"], cfg["host"], cfg["port"], cfg["db"])
        engine = get_engine(sqlalchemy_url)
        ok, msg = test_engine(engine)
        st.write(f"Engine: {'✅ OK' if ok else '❌ FAIL'}")
        if not ok:
            st.code(msg)
    except Exception as e:
        st.write("Engine: ❌ FAIL")
        st.code(str(e))
        st.stop()

# Main UI
st.subheader("Ask a question")
question = st.text_area("Example: Which country's customers spent the most by invoice?", height=90)

if st.button("Run Query"):
    if not openai_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()

    llm = get_llm(openai_key, openai_model)

    with st.spinner("Reading schema..."):
        schema = fetch_schema(engine, cfg["db"])

    with st.spinner("Generating SQL..."):
        sql = generate_sql(llm, schema, question)

    st.markdown("### Generated SQL")
    st.code(sql, language="sql")

    if not is_read_only_sql(sql):
        st.error("Blocked: generated SQL is not read-only.")
        st.stop()

    with st.spinner("Running query..."):
        df = pd.read_sql(text(sql), engine)

    st.markdown("### Results")
    st.dataframe(df, use_container_width=True)
