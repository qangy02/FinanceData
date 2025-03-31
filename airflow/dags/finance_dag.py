import requests
import pandas as pd
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()

# Configuration
# API-Key aus der Umgebungsvariable holen
API_KEY = os.getenv("FINNHUB_API_KEY")  # Finnhub API-Key
#API_KEY = "cvio5vpr01qijvgjk970cvio5vpr01qijvgjk97g"
OUTPUT_DIR = "/opt/airflow/data"
MAG7_SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]

# Logging einrichten
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. EXTRACT: Daten von Finnhub API holen
def extract():
    stock_data = []
    base_url = "https://finnhub.io/api/v1/quote"
    
    for symbol in MAG7_SYMBOLS:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or "c" not in data:
                logger.warning(f"Ungültige Daten für {symbol}: {data}")
                continue
            
            data["symbol"] = symbol
            data["timestamp"] = datetime.now().isoformat()
            stock_data.append(data)
            logger.info(f"Fetched data for {symbol}")
        
        except requests.exceptions.Timeout:
            logger.error(f"Timeout bei der Anfrage für {symbol}")
            continue
        except requests.exceptions.HTTPError as e:
            status_code = response.status_code if 'response' in locals() else 'N/A'
            logger.error(f"HTTP-Fehler für {symbol}: {e}, Status Code: {status_code}")
            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Allgemeiner Fehler bei der Anfrage für {symbol}: {e}")
            continue
    
    if not stock_data:
        logger.error("Keine Daten abgerufen. Task wird abgebrochen.")
        raise ValueError("Keine Daten von der Finnhub-API abgerufen.")
    
    logger.info(f"Erfolgreich {len(stock_data)} Symbole abgerufen.")
    return stock_data

# 2. TRANSFORM: Daten bereinigen und strukturieren
def transform(**kwargs):
    ti = kwargs["ti"]
    raw_data = ti.xcom_pull(task_ids="extract_task")
    
    if not raw_data:
        logger.error("Keine Rohdaten zum Transformieren erhalten.")
        raise ValueError("Keine Rohdaten zum Transformieren erhalten.")
    
    transformed_data = []
    for entry in raw_data:
        if not all(key in entry for key in ["c", "h", "l", "o", "pc"]):
            logger.warning(f"Fehlende Werte in den Daten für {entry['symbol']}: {entry}")
            continue
        if entry["c"] <= 0:
            logger.warning(f"Ungültiger Preis für {entry['symbol']}: {entry['c']}")
            continue
        
        # Berechne prozentuale Änderung
        percent_change = ((entry["c"] - entry["pc"]) / entry["pc"]) * 100 if entry["pc"] != 0 else 0
        
        transformed_data.append({
            "symbol": entry["symbol"],
            "current_price": entry["c"],
            "high_price": entry["h"],
            "low_price": entry["l"],
            "open_price": entry["o"],
            "previous_close": entry["pc"],
            "percent_change": percent_change,
            "timestamp": entry["timestamp"]
        })
    
    df = pd.DataFrame(transformed_data)
    df = df.dropna(subset=["current_price"])
    logger.info(f"Transformed {len(df)} stock entries after quality checks")
    
    if df.duplicated(subset=["symbol", "timestamp"]).any():
        logger.warning("Duplikate in den Daten gefunden!")
        df = df.drop_duplicates(subset=["symbol", "timestamp"])
        logger.info(f"Duplikate entfernt. Verbleibende Einträge: {len(df)}")
    
    ti.xcom_push(key="transformed_data", value=df.to_json())

# 3. LOAD: In tägliche CSV-Dateien speichern
def load(**kwargs):
    ti = kwargs["ti"]
    transformed_json = ti.xcom_pull(task_ids="transform_task", key="transformed_data")
    
    if not transformed_json:
        logger.error("Keine transformierten Daten zum Laden erhalten.")
        raise ValueError("Keine transformierten Daten zum Laden erhalten.")
    
    df = pd.read_json(transformed_json)
    
    # Tägliche CSV-Datei
    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(OUTPUT_DIR, f"mag7_stocks_{date_str}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)
    logger.info(f"Data saved to CSV: {csv_path}")

# DAG Definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "mag7_stocks_etl",
    default_args=default_args,
    description="ETL pipeline for Mag7 stocks from Finnhub API",
    schedule_interval="*/30 * * * *",  
    start_date=datetime(2025, 3, 31, 9, 0),
    catchup=False,
    is_paused_upon_creation=False,
)

# Tasks definieren
extract_task = PythonOperator(
    task_id="extract_task",
    python_callable=extract,
    dag=dag,
)

transform_task = PythonOperator(
    task_id="transform_task",
    python_callable=transform,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
    task_id="load_task",
    python_callable=load,
    provide_context=True,
    dag=dag,
)

# Task-Verkettung
extract_task >> transform_task >> load_task 