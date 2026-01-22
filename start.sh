#!/usr/bin/env bash
set -e

streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8080}" \
  --server.headless true \
  --browser.gatherUsageStats false
