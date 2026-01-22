from __future__ import annotations

from langchain_community.utilities.sql_database import SQLDatabase

from config import MySQLConfig, require_non_empty


def build_db(cfg: MySQLConfig) -> SQLDatabase:
    require_non_empty(cfg.password, "MYSQL_PASSWORD")
    return SQLDatabase.from_uri(cfg.sqlalchemy_uri())
