import requests
import pandas as pd
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

# Configuration
API_KEY = ""  # Finnhub API-Key
OUTPUT_DIR = "/opt/airflow/data"  # Airflow-typischer Pfad
CSV_PATH = os.path.join(OUTPUT_DIR, "mag7_stocks.csv")
MAG7_SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]

# Logging einrichten
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. EXTRACT: Daten von Finnhub API holen
def extract():
    stock_data = []
    
    for symbol in MAG7_SYMBOLS:
        try:
            # API-Anfrage mit Timeout (10 Sekunden)
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Wirft eine Exception bei HTTP-Fehlern
            data = response.json()
            
            # Prüfen, ob die API sinnvolle Daten zurückgibt
            if not data or "c" not in data:
                logger.warning(f"Ungültige Daten für {symbol}: {data}")
                continue
            
            # Symbol und Timestamp hinzufügen
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
    
    # Prüfen, ob Daten abgerufen wurden
    if not stock_data:
        logger.error("Keine Daten abgerufen. Task wird abgebrochen.")
        raise ValueError("Keine Daten von der Finnhub-API abgerufen.")
    
    logger.info(f"Erfolgreich {len(stock_data)} Symbole abgerufen.")
    return stock_data

# 2. TRANSFORM: Daten bereinigen und strukturieren
def transform(**kwargs):
    ti = kwargs["ti"]
    raw_data = ti.xcom_pull(task_ids="extract_task")
    
    # Prüfen, ob Daten vorhanden sind
    if not raw_data:
        logger.error("Keine Rohdaten zum Transformieren erhalten.")
        raise ValueError("Keine Rohdaten zum Transformieren erhalten.")
    
    transformed_data = []
    for entry in raw_data:
        transformed_data.append({
            "symbol": entry["symbol"],
            "current_price": entry["c"],
            "high_price": entry["h"],
            "low_price": entry["l"],
            "open_price": entry["o"],
            "previous_close": entry["pc"],
            "timestamp": entry["timestamp"]
        })
    
    df = pd.DataFrame(transformed_data)
    df = df.dropna(subset=["current_price"])  # Entferne fehlende Preise
    logger.info(f"Transformed {len(df)} stock entries")
    
    # Push transformed data to XCom
    ti.xcom_push(key="transformed_data", value=df.to_json())

# 3. LOAD: In CSV speichern
def load(**kwargs):
    ti = kwargs["ti"]
    transformed_json = ti.xcom_pull(task_ids="transform_task", key="transformed_data")
    
    # Prüfen, ob transformierte Daten vorhanden sind
    if not transformed_json:
        logger.error("Keine transformierten Daten zum Laden erhalten.")
        raise ValueError("Keine transformierten Daten zum Laden erhalten.")
    
    df = pd.read_json(transformed_json)
    
    # Verzeichnis erstellen
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # In CSV speichern
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_PATH, mode="w", header=True, index=False)
    
    logger.info(f"Data saved to CSV: {CSV_PATH}")

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
    schedule_interval="*/5 * * * *",  # Alle 5 Minuten
    start_date=datetime(2025, 3, 28, 11, 00),  # Startdatum
    catchup=False,
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