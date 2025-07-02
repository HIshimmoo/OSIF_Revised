#!/bin/sh
# Simple setup script to install Python dependencies
if command -v pip >/dev/null 2>&1; then
    pip install -r requirements.txt
else
    echo "pip not found. Please install Python pip first." >&2
    exit 1
fi
