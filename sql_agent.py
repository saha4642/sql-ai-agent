from __future__ import annotations

import argparse
from typing import Any

from langchain.agents import AgentType
from langchain_community.agent_toolkits import create_sql_agent

from config import MySQLConfig, OpenAIConfig
from db_factory import build_db
from llm_factory import build_llm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Natural Language → SQL agent for MySQL using OpenAI + LangChain."
    )
    p.add_argument("--prompt", type=str, default="", help="Natural language question to ask")
    p.add_argument("--verbose", action="store_true", help="Show agent tool calls / reasoning traces")
    p.add_argument("--show-schema", action="store_true", help="Print DB schema info and exit")
    return p.parse_args()


def _print_result(result: Any) -> None:
    # LangChain may return a dict like {"input":..., "output":...}
    if isinstance(result, dict):
        out = result.get("output") or result.get("result") or str(result)
        print(out)
    else:
        print(result)


def main() -> int:
    args = parse_args()

    llm = build_llm(OpenAIConfig())
    db = build_db(MySQLConfig())

    if args.show_schema:
        print(db.get_table_info())
        return 0

    if not args.prompt.strip():
        print('Provide a prompt with --prompt "your question"')
        return 2

    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=args.verbose,
        handle_parsing_errors=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    result = agent.invoke(args.prompt)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
