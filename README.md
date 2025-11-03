# Rayyan_Abdullah_CS22B2001
# Quant Developer Evaluation Assignment

## Overview
This is a prototype analytical app for live tick analytics using Binance WebSocket data.  
It includes:
- Real-time data ingestion & SQLite storage  
- Resampling (1s, 1m, 5m)  
- Analytics: OLS hedge ratio, spread, z-score, ADF test, rolling correlation  
- Interactive Dash frontend with alerts and CSV export  

## Run Instructions
```bash
pip install -r requirements.txt
python app.py
