#!/bin/bash
export PORT=${PORT:-8503}
streamlit run monitoring_app.py --server.port="$PORT" --server.address=0.0.0.0