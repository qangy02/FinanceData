import pandas as pd
import os
import matplotlib.pyplot as plt
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
from prophet import Prophet
import numpy as np

# Configuration
OUTPUT_DIR = "/opt/airflow/data"
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
MAG7_SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]
MINIMUM_DAYS = 20 # Mindestens 20 Tage an Daten erforderlich

# Logging einrichten
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Funktion zur Berechnung des RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 1. LOAD: Lese die CSV-Dateien für die Analyse
def load_data(**kwargs):
    ti = kwargs["ti"]
    
    all_data = []
    for file in os.listdir(OUTPUT_DIR):
        if file.startswith("mag7_stocks_") and file.endswith(".csv"):
            file_path = os.path.join(OUTPUT_DIR, file)
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    if not all_data:
        logger.error("Keine CSV-Dateien gefunden.")
        raise ValueError("Keine CSV-Dateien gefunden.")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
    
    # Prüfe, ob Daten von mindestens 20 Tagen vorliegen
    for symbol in MAG7_SYMBOLS:
        symbol_data = combined_df[combined_df["symbol"] == symbol]
        if symbol_data.empty:
            logger.warning(f"Keine Daten für {symbol} gefunden.")
            continue
        
        # Berechne die Anzahl der Tage
        min_date = symbol_data["timestamp"].min()
        max_date = symbol_data["timestamp"].max()
        days_covered = (max_date - min_date).days + 1  # +1, um den ersten Tag mit einzubeziehen
        
        if days_covered < MINIMUM_DAYS:
            logger.warning(f"Nicht genügend Daten für {symbol}: Nur {days_covered} Tage vorhanden, mindestens {MINIMUM_DAYS} Tage erforderlich.")
            ti.xcom_push(key="has_enough_data", value=False)
            return
    
    logger.info(f"Loaded {len(combined_df)} rows of data for analysis")
    ti.xcom_push(key="analysis_data", value=combined_df.to_json())
    ti.xcom_push(key="has_enough_data", value=True)

# 2. FORECAST: Zukünftige Trends vorhersagen mit Prophet
def forecast_trends(**kwargs):
    ti = kwargs["ti"]
    has_enough_data = ti.xcom_pull(task_ids="load_data_task", key="has_enough_data")
    
    if not has_enough_data:
        logger.info("Überspringe Forecast, da nicht genügend Daten vorliegen.")
        return
    
    data_json = ti.xcom_pull(task_ids="load_data_task", key="analysis_data")
    
    if not data_json:
        logger.error("Keine Daten zum Vorhersagen erhalten.")
        raise ValueError("Keine Daten zum Vorhersagen erhalten.")
    
    df = pd.read_json(data_json)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    forecast_results = {}
    forecast_plots = []
    for symbol in MAG7_SYMBOLS:
        symbol_data = df[df["symbol"] == symbol][["timestamp", "current_price"]]
        if len(symbol_data) < 2:
            logger.warning(f"Nicht genügend Daten für {symbol} zur Vorhersage.")
            continue
        
        prophet_df = symbol_data.rename(columns={"timestamp": "ds", "current_price": "y"})
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=24, freq="H")
        forecast = model.predict(future)
        
        # Berechne den prozentualen Anstieg/Fall des Preises in den nächsten 24 Stunden
        last_actual_price = prophet_df["y"].iloc[-1]
        last_forecast_price = forecast["yhat"].iloc[-1]
        price_change_percent = ((last_forecast_price - last_actual_price) / last_actual_price) * 100
        
        forecast_results[symbol] = {
            "price_change_percent": price_change_percent,
            "last_forecast_price": last_forecast_price,
            "last_actual_price": last_actual_price,
            "yhat_lower": forecast["yhat_lower"].iloc[-1],
            "yhat_upper": forecast["yhat_upper"].iloc[-1]
        }
        
        plt.figure(figsize=(12, 6))
        plt.plot(prophet_df["ds"], prophet_df["y"], label="Actual")
        plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", linestyle="--")
        plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="gray", alpha=0.2, label="Confidence Interval")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.title(f"Price Forecast for {symbol}")
        plt.legend()
        plt.xticks(rotation=45)
        # Statischer Dateiname, um die alte Datei zu überschreiben
        forecast_plot_path = os.path.join(ANALYSIS_DIR, f"forecast_{symbol}.png")
        plt.savefig(forecast_plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Forecast plot for {symbol} saved to {forecast_plot_path}")
        forecast_plots.append(forecast_plot_path)
    
    ti.xcom_push(key="forecast_results", value=forecast_results)
    ti.xcom_push(key="forecast_plots", value=forecast_plots)

# 3. ANALYZE: Berechne Indikatoren, Trading-Signale und erstelle Visualisierungen
def analyze_and_visualize(**kwargs):
    ti = kwargs["ti"]
    has_enough_data = ti.xcom_pull(task_ids="load_data_task", key="has_enough_data")
    
    if not has_enough_data:
        logger.info("Überspringe Analyse, da nicht genügend Daten vorliegen.")
        return
    
    data_json = ti.xcom_pull(task_ids="load_data_task", key="analysis_data")
    forecast_results = ti.xcom_pull(task_ids="forecast_task", key="forecast_results")
    
    if not data_json:
        logger.error("Keine Daten zum Analysieren erhalten.")
        raise ValueError("Keine Daten zum Analysieren erhalten.")
    
    if not forecast_results:
        logger.error("Keine Vorhersagen von Prophet erhalten.")
        raise ValueError("Keine Vorhersagen von Prophet erhalten.")
    
    df = pd.read_json(data_json)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # Trading-Signale für jedes Symbol berechnen
    trading_signals = []
    for symbol in MAG7_SYMBOLS:
        symbol_data = df[df["symbol"] == symbol].copy()
        symbol_data = symbol_data.sort_values("timestamp")
        
        if len(symbol_data) < 50:  # Mindestens 50 Datenpunkte für MA50
            logger.warning(f"Nicht genügend Datenpunkte für {symbol} zur Berechnung der Indikatoren.")
            continue
        
        # Indikator 1: Gleitende Durchschnitte
        symbol_data["ma20"] = symbol_data["current_price"].rolling(window=20).mean()
        symbol_data["ma50"] = symbol_data["current_price"].rolling(window=50).mean()
        symbol_data["ma_signal"] = 0
        symbol_data["ma_signal"] = np.where(
            (symbol_data["ma20"] > symbol_data["ma50"]) & (symbol_data["ma20"].shift(1) <= symbol_data["ma50"].shift(1)), 1, 0
        )  # Golden Cross: Kauf
        symbol_data["ma_signal"] = np.where(
            (symbol_data["ma20"] < symbol_data["ma50"]) & (symbol_data["ma20"].shift(1) >= symbol_data["ma50"].shift(1)), -1, symbol_data["ma_signal"]
        )  # Death Cross: Verkauf
        
        # Indikator 2: RSI
        symbol_data["rsi"] = calculate_rsi(symbol_data["current_price"], periods=14)
        symbol_data["rsi_signal"] = 0
        symbol_data["rsi_signal"] = np.where(symbol_data["rsi"] < 30, 1, 0)  # Überverkauft: Kauf
        symbol_data["rsi_signal"] = np.where(symbol_data["rsi"] > 70, -1, symbol_data["rsi_signal"])  # Überkauft: Verkauf
        
        # Indikator 3: Bollinger Bands
        symbol_data["bb_middle"] = symbol_data["current_price"].rolling(window=20).mean()
        symbol_data["bb_std"] = symbol_data["current_price"].rolling(window=20).std()
        symbol_data["bb_upper"] = symbol_data["bb_middle"] + 2 * symbol_data["bb_std"]
        symbol_data["bb_lower"] = symbol_data["bb_middle"] - 2 * symbol_data["bb_std"]
        symbol_data["bb_signal"] = 0
        symbol_data["bb_signal"] = np.where(symbol_data["current_price"] < symbol_data["bb_lower"], 1, 0)  # Unter dem unteren Band: Kauf
        symbol_data["bb_signal"] = np.where(symbol_data["current_price"] > symbol_data["bb_upper"], -1, symbol_data["bb_signal"])  # Über dem oberen Band: Verkauf
        
        # Indikator 4: Prophet-Vorhersage
        symbol_data["prophet_signal"] = 0
        if symbol in forecast_results:
            price_change_percent = forecast_results[symbol]["price_change_percent"]
            symbol_data["prophet_signal"] = np.where(price_change_percent > 2, 1, 0)  # Aufwärtstrend: Kauf
            symbol_data["prophet_signal"] = np.where(price_change_percent < -2, -1, symbol_data["prophet_signal"])  # Abwärtstrend: Verkauf
        
        # Kombiniere alle Signale: Nur wenn alle Indikatoren übereinstimmen
        symbol_data["final_signal"] = 0
        symbol_data["final_signal"] = np.where(
            (symbol_data["ma_signal"] == 1) & (symbol_data["rsi_signal"] == 1) & 
            (symbol_data["bb_signal"] == 1) & (symbol_data["prophet_signal"] == 1), 1, symbol_data["final_signal"]
        )  # Kauf: Alle Indikatoren zeigen Kauf
        symbol_data["final_signal"] = np.where(
            (symbol_data["ma_signal"] == -1) & (symbol_data["rsi_signal"] == -1) & 
            (symbol_data["bb_signal"] == -1) & (symbol_data["prophet_signal"] == -1), -1, symbol_data["final_signal"]
        )  # Verkauf: Alle Indikatoren zeigen Verkauf
        
        # Berechne Entry, Stop-Loss und Take-Profit für jedes Signal
        symbol_data["entry_price"] = np.nan
        symbol_data["stop_loss"] = np.nan
        symbol_data["take_profit"] = np.nan
        
        for i in range(1, len(symbol_data)):
            if symbol_data["final_signal"].iloc[i] == 1:  # Kauf
                entry_price = symbol_data["current_price"].iloc[i]
                symbol_data["entry_price"].iloc[i] = entry_price
                recent_low = symbol_data["low_price"].iloc[max(0, i-10):i].min()
                symbol_data["stop_loss"].iloc[i] = recent_low
                risk = entry_price - recent_low
                symbol_data["take_profit"].iloc[i] = entry_price + 2 * risk  # 2:1 Risk-to-Reward
            elif symbol_data["final_signal"].iloc[i] == -1:  # Verkauf
                entry_price = symbol_data["current_price"].iloc[i]
                symbol_data["entry_price"].iloc[i] = entry_price
                recent_high = symbol_data["high_price"].iloc[max(0, i-10):i].max()
                symbol_data["stop_loss"].iloc[i] = recent_high
                risk = recent_high - entry_price
                symbol_data["take_profit"].iloc[i] = entry_price - 2 * risk  # 2:1 Risk-to-Reward
        
        # Backtesting: Simuliere Trades und berechne Gewinne/Verluste
        symbol_data["trade_result"] = np.nan
        symbol_data["profit_loss"] = np.nan
        active_trade = None
        for i in range(len(symbol_data)):
            if symbol_data["final_signal"].iloc[i] == 1:  # Kauf
                active_trade = {
                    "entry_price": symbol_data["entry_price"].iloc[i],
                    "stop_loss": symbol_data["stop_loss"].iloc[i],
                    "take_profit": symbol_data["take_profit"].iloc[i],
                    "entry_index": i
                }
            elif symbol_data["final_signal"].iloc[i] == -1 and active_trade:  # Verkauf (Schließe Trade)
                exit_price = symbol_data["current_price"].iloc[i]
                profit_loss = exit_price - active_trade["entry_price"]
                symbol_data["profit_loss"].iloc[i] = profit_loss
                symbol_data["trade_result"].iloc[i] = 1 if profit_loss > 0 else -1
                active_trade = None
            elif active_trade:  # Prüfe Stop-Loss und Take-Profit
                current_price = symbol_data["current_price"].iloc[i]
                if current_price <= active_trade["stop_loss"]:
                    profit_loss = active_trade["stop_loss"] - active_trade["entry_price"]
                    symbol_data["profit_loss"].iloc[active_trade["entry_index"]] = profit_loss
                    symbol_data["trade_result"].iloc[active_trade["entry_index"]] = -1
                    active_trade = None
                elif current_price >= active_trade["take_profit"]:
                    profit_loss = active_trade["take_profit"] - active_trade["entry_price"]
                    symbol_data["profit_loss"].iloc[active_trade["entry_index"]] = profit_loss
                    symbol_data["trade_result"].iloc[active_trade["entry_index"]] = 1
                    active_trade = None
        
        # Visualisierung: Aktienpreis, Indikatoren und Trading-Signale
        plt.figure(figsize=(12, 6))
        plt.plot(symbol_data["timestamp"], symbol_data["current_price"], label="Price", color="blue")
        plt.plot(symbol_data["timestamp"], symbol_data["ma20"], label="MA20", color="orange")
        plt.plot(symbol_data["timestamp"], symbol_data["ma50"], label="MA50", color="green")
        plt.plot(symbol_data["timestamp"], symbol_data["bb_upper"], label="BB Upper", color="red", linestyle="--")
        plt.plot(symbol_data["timestamp"], symbol_data["bb_lower"], label="BB Lower", color="red", linestyle="--")
        
        # Markiere Kauf- und Verkauf-Signale
        buy_signals = symbol_data[symbol_data["final_signal"] == 1]
        sell_signals = symbol_data[symbol_data["final_signal"] == -1]
        plt.scatter(buy_signals["timestamp"], buy_signals["entry_price"], marker="^", color="green", label="Buy", s=100)
        plt.scatter(sell_signals["timestamp"], sell_signals["entry_price"], marker="v", color="red", label="Sell", s=100)
        
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.title(f"Trading Strategy for {symbol} (Combined Indicators)")
        plt.legend()
        plt.xticks(rotation=45)
        # Statischer Dateiname, um die alte Datei zu überschreiben
        trading_plot_path = os.path.join(ANALYSIS_DIR, f"trading_strategy_{symbol}.png")
        plt.savefig(trading_plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Trading strategy plot for {symbol} saved to {trading_plot_path}")
        
        # Speichere die Signale (nur wenn ein Signal vorhanden ist)
        signals_df = symbol_data[symbol_data["final_signal"] != 0][[
            "timestamp", "symbol", "current_price", "ma20", "ma50", "rsi", 
            "bb_upper", "bb_lower", "final_signal", "entry_price", "stop_loss", "take_profit", "profit_loss", "trade_result"
        ]]
        if not signals_df.empty:
            trading_signals.append(signals_df)
    
    # Kombiniere alle Signale in einen DataFrame und speichere sie
    if trading_signals:
        trading_signals_df = pd.concat(trading_signals, ignore_index=True)
        trading_signals_path = os.path.join(ANALYSIS_DIR, f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        trading_signals_df.to_csv(trading_signals_path, index=False)
        logger.info(f"Trading signals saved to {trading_signals_path}")
        
        # Backtesting-Ergebnisse zusammenfassen
        total_trades = len(trading_signals_df[trading_signals_df["trade_result"].notna()])
        winning_trades = len(trading_signals_df[trading_signals_df["trade_result"] == 1])
        losing_trades = len(trading_signals_df[trading_signals_df["trade_result"] == -1])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = trading_signals_df["profit_loss"].sum()
        logger.info(f"Backtesting Results: Total Trades: {total_trades}, Win Rate: {win_rate:.2f}%, Total Profit: {total_profit:.2f}")
    else:
        logger.info("Keine Trading-Signale generiert, da nicht alle Indikatoren übereinstimmen.")

# 4. SAVE RESULTS: Speichere zusätzliche Analyseergebnisse als CSV
def save_analysis_results(**kwargs):
    ti = kwargs["ti"]
    has_enough_data = ti.xcom_pull(task_ids="load_data_task", key="has_enough_data")
    
    if not has_enough_data:
        logger.info("Überspringe Speichern der Analyseergebnisse, da nicht genügend Daten vorliegen.")
        return
    
    data_json = ti.xcom_pull(task_ids="load_data_task", key="analysis_data")
    
    if not data_json:
        logger.error("Keine Daten zum Speichern der Analyseergebnisse erhalten.")
        raise ValueError("Keine Daten zum Speichern der Analyseergebnisse erhalten.")
    
    df = pd.read_json(data_json)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    summary = df.groupby("symbol").agg({
        "current_price": ["mean", "std"],
        "percent_change": ["mean", "std"]
    }).reset_index()
    summary.columns = ["symbol", "avg_price", "price_std", "avg_percent_change", "percent_change_std"]
    
    summary_path = os.path.join(ANALYSIS_DIR, f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Analysis summary saved to {summary_path}")

# DAG Definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "mag7_stocks_analysis",
    default_args=default_args,
    description="Analysis pipeline for Mag7 stocks data",
    schedule_interval="@daily",  # Läuft jede 24h
    start_date=datetime(2025, 3, 31, 9, 0),
    catchup=False,
    is_paused_upon_creation=False,  
)


# Tasks definieren
load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

forecast_task = PythonOperator(
    task_id="forecast_task",
    python_callable=forecast_trends,
    provide_context=True,
    dag=dag,
)

analyze_task = PythonOperator(
    task_id="analyze_task",
    python_callable=analyze_and_visualize,
    provide_context=True,
    dag=dag,
)

save_results_task = PythonOperator(
    task_id="save_results_task",
    python_callable=save_analysis_results,
    provide_context=True,
    dag=dag,
)

# Task-Verkettung
load_data_task >> forecast_task >> analyze_task >> save_results_task