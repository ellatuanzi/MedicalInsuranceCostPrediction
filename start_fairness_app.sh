#!/bin/bash
export PORT=${PORT:-8502}
streamlit run fairness_app.py --server.port="$PORT" --server.address=0.0.0.0