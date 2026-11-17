#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
(sleep 2 && open "http://127.0.0.1:8000/setup" 2>/dev/null || xdg-open "http://127.0.0.1:8000/setup" 2>/dev/null) &
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
