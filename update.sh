#!/bin/bash
# Tennis Predictor Daily Update
# Run manually: ./update.sh
# Or add to system crontab: crontab -e  → add:  7 6 * * * /Users/pipemassuh/tennis-predictor/update.sh >> /tmp/tennis_update.log 2>&1

set -e
cd /Users/pipemassuh/tennis-predictor

echo "=== Tennis Predictor Update $(date) ==="

# Run the full pipeline
python -m src.ingestion.daily_update

# Restart dashboard
pkill -f "streamlit run" 2>/dev/null || true
sleep 2
streamlit run src/dashboard/app.py --server.headless true &>/tmp/streamlit.log &
echo "Dashboard restarted at localhost:8501"
echo "Done."
