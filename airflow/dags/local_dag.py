import requests
import pandas as pd
import os
import sqlite3
from datetime import datetime
import logging

# Configuration
API_KEY = ""  # Finnhub API-Key
OUTPUT_DIR = "./data"  # Lokaler Pfad für die Ausgabe
CSV_PATH = os.path.join(OUTPUT_DIR, "mag7_stocks.csv")
DB_PATH = os.path.join(OUTPUT_DIR, "mag7_stocks.db")
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
            logger.error(f"HTTP-Fehler für {symbol}: {e}, Status Code: {response.status_code}")
            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Allgemeiner Fehler bei der Anfrage für {symbol}: {e}")
            continue
    
    # Prüfen, ob Daten abgerufen wurden
    if not stock_data:
        logger.error("Keine Daten abgerufen. Programm wird abgebrochen.")
        raise ValueError("Keine Daten von der Finnhub-API abgerufen.")
    
    logger.info(f"Erfolgreich {len(stock_data)} Symbole abgerufen.")
    return stock_data

# 2. TRANSFORM: Daten bereinigen und strukturieren
def transform(raw_data):
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
    return df

# 3. LOAD: In CSV und SQLite speichern
def load(df):
    # Verzeichnis erstellen
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # In CSV speichern
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_PATH, mode="w", header=True, index=False)
    
    # In SQLite speichern
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("mag7_stocks", conn, if_exists="append", index=False)
    conn.close()
    
    logger.info(f"Data saved to CSV: {CSV_PATH} and SQLite: {DB_PATH}")

# Hauptfunktion zum Ausführen der ETL-Pipeline
def run_etl():
    try:
        # Schritt 1: Extract
        raw_data = extract()
        
        # Schritt 2: Transform
        transformed_df = transform(raw_data)
        
        # Schritt 3: Load
        load(transformed_df)
        
        logger.info("ETL-Pipeline erfolgreich abgeschlossen.")
    
    except Exception as e:
        logger.error(f"Fehler in der ETL-Pipeline: {e}")
        raise

# Skript ausführen
if __name__ == "__main__":
    run_etl()