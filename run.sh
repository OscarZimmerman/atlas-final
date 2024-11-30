#!/bin/bash
set -e

echo "Starting analysis..."
python /app/analysis/process.py
echo "Analysis complete. Output saved to /app/output/plot.png."
