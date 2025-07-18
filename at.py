import os
import json
import requests
import sqlite3
from datetime import datetime, timedelta
import python_bithumb
import google.generativeai as genai
import pandas as pd
import pandas_ta as ta # pandas_ta 라이브러리 설치 필요: pip install pandas_ta
import time
import sys
import traceback
from decimal import Decimal
from numpy import nan as npNaN

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QListWidget, QListWidgetItem, QMessageBox, QSizePolicy,
    QAbstractItemView, QLineEdit, QDialog, QGroupBox, QRadioButton
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QScreen, QDoubleValidator

import pyqtgraph as pg

# --- 코인 심볼 설정 ---
TRADE_SETTINGS_FILE = "trade_settings.json"

def _load_trade_settings():
    """트레이딩 설정을 파일에서 로드합니다. 파일이 없거나 유효하지 않으면 기본값을 반환합니다."""
    try:
        if os.path.exists(TRADE_SETTINGS_FILE):
            with open(TRADE_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                return settings.get("trade_coin_symbol", "BTC")
    except Exception as e:
        print(f"Error loading trade settings: {e}. Using default BTC.")
    return "BTC"

def _save_trade_settings(symbol):
    """트레이딩 설정을 파일에 저장합니다."""
    try:
        with open(TRADE_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"trade_coin_symbol": symbol}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving trade settings: {e}")

# 시작 시점에 TRADE_COIN_SYMBOL을 설정 파일에서 로드
TRADE_COIN_SYMBOL = _load_trade_settings()

# --- API 키 로드/저장 함수 ---
API_KEYS_FILE = "api_keys.json" # API 키를 저장할 파일

def _load_api_keys():
    """API 키를 파일에서 로드합니다. 파일이 없거나 유효하지 않으면 빈 딕셔너리를 반환합니다."""
    keys = {
        "bithumb_access_key": "",
        "bithumb_secret_key": "",
        "serpapi_api_key": "",
        "gemini_api_key": ""
    }
    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
                loaded_keys = json.load(f)
                keys.update(loaded_keys)
                print("API keys loaded successfully.")
    except Exception as e:
        print(f"Error loading API keys: {e}. Using empty keys.")
    return keys

def _save_api_keys(new_keys):
    """API 키를 파일에 저장합니다."""
    try:
        with open(API_KEYS_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_keys, f, ensure_ascii=False, indent=4)
        print("API keys saved successfully.")
    except Exception as e:
        print(f"Error saving API keys: {e}")

# 시작 시점에 API 키 로드
_current_api_keys = _load_api_keys()
BITHUMB_ACCESS_KEY = _current_api_keys.get("bithumb_access_key", "")
BITHUMB_SECRET_KEY = _current_api_keys.get("bithumb_secret_key", "")
SERPAPI_API_KEY = _current_api_keys.get("serpapi_api_key", "")
GEMINI_API_KEY = _current_api_keys.get("gemini_api_key", "")

# --- 백엔드 로직 ---

# SQLite 데이터베이스 초기화 함수
def init_db():
    conn = sqlite3.connect(f"{TRADE_COIN_SYMBOL.lower()}_trading.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  decision TEXT,
                  percentage INTEGER,
                  reason TEXT,
                  coin_symbol TEXT,
                  coin_balance REAL,
                  krw_balance REAL,
                  coin_price REAL)''')

    # Check if coin_symbol column exists, if not, add it
    c.execute("PRAGMA table_info(trades)")
    columns = [col[1] for col in c.fetchall()]

    if 'coin_symbol' not in columns:
        c.execute("ALTER TABLE trades ADD COLUMN coin_symbol TEXT")
        print("Added 'coin_symbol' column to 'trades' table.")

    if 'coin_balance' not in columns:
        c.execute("ALTER TABLE trades ADD COLUMN coin_balance REAL")
        print("Added 'coin_balance' column to 'trades' table.")
    if 'krw_balance' not in columns:
        c.execute("ALTER TABLE trades ADD COLUMN krw_balance REAL")
        print("Added 'krw_balance' column to 'trades' table.")
    if 'coin_price' not in columns:
        c.execute("ALTER TABLE trades ADD COLUMN coin_price REAL")
        print("Added 'coin_price' column to 'trades' table.")

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


# 거래 정보를 DB에 기록하는 함수
def log_trade(conn, decision, percentage, reason, coin_symbol, coin_balance, krw_balance, coin_price):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT INTO trades
                 (timestamp, decision, percentage, reason, coin_symbol, coin_balance, krw_balance, coin_price)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (timestamp, decision, percentage, reason, coin_symbol, coin_balance, krw_balance, coin_price))
    conn.commit()


# DB 연결 가져오기
def get_db_connection():
    return sqlite3.connect(f"{TRADE_COIN_SYMBOL.lower()}_trading.db")

def get_simple_average_purchase_price(coin_symbol):
    """
    지정된 코인 심볼의 단순 평균 매수 단가를 계산하여 반환합니다.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    SELECT coin_balance, coin_price FROM trades
    WHERE decision = 'buy' AND coin_symbol = ?
    ORDER BY timestamp ASC
    """, (coin_symbol,))
    buy_trades = c.fetchall()
    conn.close()

    total_cost = 0
    total_bought_amount = 0

    for amount, price in buy_trades:
        total_cost += amount * price
        total_bought_amount += amount

    if total_bought_amount > 0:
        return total_cost / total_bought_amount
    else:
        return 0 # 매수 내역이 없는 경우 0 반환

# 최근 거래 내역 가져오기 (대시보드 테이블용)
def get_recent_trades_data(limit=5):
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("""
    SELECT timestamp, decision, percentage, reason, coin_symbol, coin_balance, krw_balance, coin_price
    FROM trades
    ORDER BY timestamp DESC
    LIMIT ?
    """, (limit,))

    rows = c.fetchall()
    print(f"DEBUG: get_recent_trades_data - Fetched raw rows from DB: {rows}")

    columns = ['timestamp', 'decision', 'percentage', 'reason', 'coin_symbol', 'coin_balance', 'krw_balance',
               'coin_price']
    trades = []

    for row in rows:
        trade = {columns[i]: row[i] for i in range(len(columns))}
        trades.append(trade)

    print(f"DEBUG: get_recent_trades_data - Processed trades list: {trades}")

    conn.close()
    return trades


# 모든 거래 내역 가져오기 (그래프용, 시간 오름차순)
def get_all_trades_for_graph():
    conn = get_db_connection()
    c = conn.cursor()

    two_days_ago = datetime.now() - timedelta(days=2) # 2일 데이터만 가져옴
    two_days_ago_iso = two_days_ago.isoformat()

    c.execute(f"""
    SELECT timestamp, decision, percentage, reason, coin_symbol, coin_balance, krw_balance, coin_price
    FROM trades
    WHERE timestamp >= '{two_days_ago_iso}'
    ORDER BY timestamp ASC
    """)
    columns = ['timestamp', 'decision', 'percentage', 'reason', 'coin_symbol', 'coin_balance', 'krw_balance',
               'coin_price']
    trades = []
    for row in c.fetchall():
        trade = {columns[i]: row[i] for i in range(len(columns))}
        trades.append(trade)
    conn.close()
    return trades


# 뉴스 데이터 가져오는 함수 (SerpApi 사용)
def get_crypto_news_data(api_key, query_symbol, location="us", language="en", num_results=5):
    params = {
        "engine": "google_news", "q": f"{query_symbol} news", "gl": location,
        "hl": language, "api_key": api_key
    }
    api_url = "https://serpapi.com/search.json"
    news_data = []

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        results = response.json()

        if "news_results" in results:
            for news_item in results["news_results"][:num_results]:
                news_data.append({
                    "title": news_item.get("title"),
                    "date": news_item.get("date")
                })
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
    return news_data


# 빗썸 API에서 현재 가격을 가져오는 함수
def get_current_price_from_bithumb_api(symbol):
    url = f"https://api.bithumb.com/public/ticker/{symbol}_KRW"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        status = data.get("status")

        if status == "0000":
            coin_data = data.get("data", {})
            closing_price = Decimal(coin_data.get("closing_price", "0"))
            return float(closing_price)
        else:
            print(f"ERROR: Bithumb {symbol} ticker API call failed: Status code {status}")
            return None
    except requests.RequestException as e:
        print(f"ERROR: HTTP request error fetching {symbol} price: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"ERROR: Data processing error fetching {symbol} price: {e}")
        traceback.print_exc()
        return None


def _fetch_candlestick_data_direct(symbol, interval_str, count):
    """
    빗썸 공개 API를 직접 호출하여 캔들스틱(OHLCV) 데이터를 가져오는 헬퍼 함수.
    """
    bithumb_interval_map = {
        "minute5": "5m",
        "minute60": "1h",
        "minute240": "6h",
        "day": "24h"
    }

    api_interval = bithumb_interval_map.get(interval_str)
    if not api_interval:
        print(f"오류: 지원하지 않는 인터벌 형식입니다: {interval_str}")
        return None

    url = f"https://api.bithumb.com/public/candlestick/{symbol}_KRW/{api_interval}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "0000":
            candles = data.get("data", [])
            if not candles:
                print(f"경고: {symbol} ({interval_str}) 캔들스틱 데이터가 비어 있습니다.")
                return pd.DataFrame()

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            numeric_cols = ['open', 'close', 'high', 'low', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df.tail(count)
        else:
            print(f"오류: 빗썸 API 응답 상태 코드: {data.get('status')} - {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"HTTP 요청 오류: {e}")
        return None
    except json.JSONDecodeError:
        print(f"JSON 디코딩 오류: API 응답이 유효한 JSON 형식이 아닙니다. 응답: {response.text[:200]}...")
        return None
    except Exception as e:
        print(f"데이터 처리 중 예기치 않은 오류 발생: {e}")
        traceback.print_exc()
        return None


def calculate_stoch_rsi(df: pd.DataFrame, rsi_window: int = 14, stoch_window: int = 14, k_window: int = 3,
                        d_window: int = 3) -> pd.DataFrame:
    """
    DataFrame에 Stochastic RSI 지표를 추가합니다.
    Args:
        df (pd.DataFrame): OHLCV 데이터를 포함하는 DataFrame. 'close' 컬럼이 있어야 합니다.
        rsi_window (int): RSI 계산에 사용할 기간. 기본값은 14.
        stoch_window (int): Stochastic RSI 계산에 사용할 기간. 기본값은 14.
        k_window (int): %K 라인에 사용할 이동 평균 기간. 기본값은 3.
        d_window (int): %D 라인에 사용할 이동 평균 기간. 기본값은 3.
    Returns:
        pd.DataFrame: Stochastic RSI (%K, %D) 컬럼이 추가된 DataFrame.
    """
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)

    # Calculate Stochastic RSI using pandas_ta
    df.ta.stochrsi(close='close', length=rsi_window, rsi_length=stoch_window, k=k_window, d=d_window, append=True)
    return df

# AI 트레이딩 로직 함수
def ai_trading_logic():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Starting AI trading logic for {TRADE_COIN_SYMBOL}...")

    # 모든 DataFrame 변수를 None으로 초기화합니다.
    very_short_term_df = None
    short_term_df = None
    mid_term_df = None
    long_term_df = None

    # 1. 빗썸 차트 데이터 수집 (직접 API 호출 함수 사용)
    very_short_term_df = _fetch_candlestick_data_direct(TRADE_COIN_SYMBOL, interval_str="minute5", count=200)
    if very_short_term_df is not None and not very_short_term_df.empty:
        very_short_term_df = calculate_stoch_rsi(very_short_term_df.copy())
    else:
        print("DEBUG: very_short_term_df (5m) is None or empty. Skipping StochRSI calculation.")


    short_term_df = _fetch_candlestick_data_direct(TRADE_COIN_SYMBOL, interval_str="minute60", count=100)
    if short_term_df is not None and not short_term_df.empty:
        short_term_df = calculate_stoch_rsi(short_term_df.copy())
    else:
        print("DEBUG: short_term_df (60m) is None or empty. Skipping StochRSI calculation.")


    mid_term_df = _fetch_candlestick_data_direct(TRADE_COIN_SYMBOL, interval_str="minute240", count=50)
    if mid_term_df is not None and not mid_term_df.empty:
        mid_term_df = calculate_stoch_rsi(mid_term_df.copy())
    else:
        print("DEBUG: mid_term_df (240m) is None or empty. Skipping StochRSI calculation.")


    long_term_df = _fetch_candlestick_data_direct(TRADE_COIN_SYMBOL, interval_str="day", count=60)
    if long_term_df is not None and not long_term_df.empty:
        long_term_df = calculate_stoch_rsi(long_term_df.copy())
    else:
        print("DEBUG: long_term_df (day) is None or empty. Skipping StochRSI calculation.")

    # 모든 DataFrame이 비어있으면 로직을 더 이상 진행하지 않음
    if (very_short_term_df is None or very_short_term_df.empty) and \
       (short_term_df is None or short_term_df.empty) and \
       (mid_term_df is None or mid_term_df.empty) and \
       (long_term_df is None or long_term_df.empty):
        print("ERROR: All DataFrames are empty. Cannot proceed with trading logic.")
        return {"decision": "hold", "percentage": 0, "reason": "All DataFrames are empty."}

    # 2. 뉴스 데이터 수집
    news_articles = []
    if SERPAPI_API_KEY:
        try:
            news_articles = get_crypto_news_data(SERPAPI_API_KEY, TRADE_COIN_SYMBOL, "us", "en", 5)
            print(f"DEBUG: Fetched {len(news_articles)} news articles.")
        except Exception as e:
            print(f"Error fetching news data: {e}")
            news_articles = []
    else:
        print("SerpApi API key not configured. Skipping news data collection.")

    # 3. 빗썸 API 연결
    if not BITHUMB_ACCESS_KEY or not BITHUMB_SECRET_KEY:
        print("BITHUMB_ACCESS_KEY 또는 BITHUMB_SECRET_KEY가 설정되지 않았습니다. API 키 설정을 확인하세요.")
        return {"decision": "hold", "percentage": 0, "reason": "Bithumb API keys not configured"}

    try:
        bithumb = python_bithumb.Bithumb(BITHUMB_ACCESS_KEY, BITHUMB_SECRET_KEY)
    except Exception as e:
        print(f"Error connecting to Bithumb API: {e}")
        return {"decision": "hold", "percentage": 0, "reason": f"Bithumb API connection failed: {e}"}

    # 4. 현재 잔고 확인
    try:
        my_krw = bithumb.get_balance("KRW")
        my_coin = bithumb.get_balance(TRADE_COIN_SYMBOL)
    except Exception as e:
        print(f"Error fetching balance from Bithumb: {e}")
        my_krw = 0
        my_coin = 0

    # 코인 현재 가격을 직접 API 호출로 가져옴
    raw_current_coin_price = get_current_price_from_bithumb_api(TRADE_COIN_SYMBOL)
    print(f"DEBUG: Raw {TRADE_COIN_SYMBOL} price in ai_trading_logic: {raw_current_coin_price}")
    current_coin_price = raw_current_coin_price if raw_current_coin_price is not None else 0.0

    # 5. 최근 거래 내역 가져오기
    try:
        recent_trades = get_recent_trades_data(limit=5)
        print(f"DEBUG: Fetched {len(recent_trades)} recent trades.")
    except Exception as e:
        print(f"Error fetching recent trades data: {e}")
        recent_trades = []

    # 6. 데이터 페이로드 준비
    data_payload = {
        "very_short_term": json.loads(very_short_term_df.to_json(orient='records')) if very_short_term_df is not None else None,
        "short_term": json.loads(short_term_df.to_json(orient='records')) if short_term_df is not None else None,
        "mid_term": json.loads(mid_term_df.to_json(orient='records')) if mid_term_df is not None else None,
        "long_term": json.loads(long_term_df.to_json(orient='records')) if long_term_df is not None else None,
        "news": news_articles,
        "current_balance": {
            "krw": my_krw,
            "coin_symbol": TRADE_COIN_SYMBOL,
            "coin_balance": my_coin,
            "coin_price": current_coin_price,
            "total_value": (my_krw + (my_coin * current_coin_price)) if isinstance(my_krw, (int, float)) and isinstance(
                my_coin, (int, float)) and isinstance(current_coin_price, (int, float)) else "N/A"
        },
        "recent_trades": recent_trades
    }

    # 7. Gemini API에게 판단 요청
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY가 설정되지 않았습니다. API 키 설정을 확인하세요.")
        return {"decision": "hold", "percentage": 0, "reason": "Gemini API key not configured"}

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

    system_instruction = (
        f"You are an expert in {TRADE_COIN_SYMBOL} investing. "
        "You invest according to the following principles:\n"
        "Rule No.1: Don't lose money as you can.\n"
        "Rule No.2: Please consider how to invest safely without investing too much money considering the balance of your holdings\n\n"
        "Analyze the provided data:\n"
        "1. **Chart Data:** Multi-timeframe OHLCV (Open, High, Low, Close, Volume) data. "
        "EACH CANDLESTICK ALSO INCLUDES STOCHASTIC RSI (%K and %D lines) VALUES. "
        "('very_short_term': 5m, 'short_term': 1h, 'mid_term': 4h, 'long_term': daily).\n"
        "2. **News Data:** Recent crypto news articles with 'title' and 'date'.\n"
        "3. **Current Balance:** Current KRW and coin balances and current coin price.\n"
        "4. **Recent Trades:** History of recent trading decisions and their outcomes.\n\n"
        "When analyzing recent trades:\n"
        "- Evaluate if previous decisions were profitable\n"
        "- Check if market conditions have changed since the last trade\n"
        "- Consider how the market reacted to your previous decisions\n"
        "- Learn from successful and unsuccessful trades\n"
        "- Maintain consistency in your strategy unless there's a clear reason to change\n\n"
        f"**Task:** Based on technical analysis (including volume and Stochastic RSI), news sentiment, and trading history, "
        f"decide whether to **buy**, **sell**, or **hold** {TRADE_COIN_SYMBOL}.\n"
        "For buy or sell decisions, include a percentage (1-100) indicating what portion of available funds to use.\n\n"
        "When a buying decision is made, the buying criteria will be converted to points, and the buying will only proceed if the buying score is above 60\n"
        "**Consider the Stochastic RSI levels:** "
        "Stochastic RSI values range from 0 to 100. "
        "Typically, values above 80 indicate overbought conditions (potential for sell-off), "
        "and values below 20 indicate oversold conditions (potential for rebound). "
        "Pay attention to **crosses between the %K and %D lines** (bullish %K crossing above %D, bearish %K crossing below %D) "
        "and **divergences** between Stochastic RSI and price. "
        "Also, look for the %K and %D lines exiting overbought/oversold zones.\n\n"
        "**Output Format:** Respond ONLY in JSON format like:\n"
        "{\"decision\": \"buy\", \"percentage\": 20, \"reason\": \"some technical reason\"}\n"
        "{\"decision\": \"sell\", \"percentage\": 50, \"reason\": \"some technical reason\"}\n"
        "{\"decision\": \"hold\", \"percentage\": 0, \"reason\": \"some technical reason\"}"
    )

    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [{"text": system_instruction}]},
                {"role": "model", "parts": [
                    {"text": "Okay, I understand. Please provide the chart, news, balance, and trade history data."}]},
                {"role": "user", "parts": [{"text": json.dumps(data_payload)}]}
            ],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "decision": {"type": "STRING", "enum": ["buy", "sell", "hold"]},
                        "percentage": {"type": "INTEGER"},
                        "reason": {"type": "STRING"}
                    },
                    "required": ["decision", "percentage", "reason"]
                }
            )
        )

        raw_result = response.text
        result = json.loads(raw_result)
        print(f"DEBUG: Gemini response: {result}")

    except json.JSONDecodeError as e:
        print(f"JSON decoding error from Gemini API: {e}. Raw response: {raw_result}")
        return {"decision": "hold", "percentage": 0, "reason": f"JSON decoding error: {e}"}
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {"decision": "hold", "percentage": 0, "reason": f"Gemini API call failed: {e}"}

    return result


# 트레이딩 실행 함수 (백엔드에서 호출)
def execute_trade_logic():
    conn = get_db_connection()

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time_str}] 트레이딩 작업 실행 중...")

    result = ai_trading_logic()
    print(f"AI Raw Result: {result}")

    if not BITHUMB_ACCESS_KEY or not BITHUMB_SECRET_KEY:
        print("BITHUMB_ACCESS_KEY 또는 BITHUMB_SECRET_KEY가 설정되지 않았습니다. API 키 설정을 확인하세요.")
        return {"decision": "hold", "percentage": 0, "reason": "Bithumb API keys not configured", "updated_krw": 0, "updated_coin": 0, "updated_coin_price": 0}

    bithumb = python_bithumb.Bithumb(BITHUMB_ACCESS_KEY, BITHUMB_SECRET_KEY)

    my_krw = bithumb.get_balance("KRW")
    my_coin = bithumb.get_balance(TRADE_COIN_SYMBOL)

    raw_current_coin_price = get_current_price_from_bithumb_api(TRADE_COIN_SYMBOL)
    print(f"DEBUG: Raw {TRADE_COIN_SYMBOL} price in execute_trade_logic: {raw_current_coin_price}")
    current_coin_price = raw_current_coin_price if raw_current_coin_price is not None else 0.0

    print(f"### AI Decision: {result['decision'].upper()} ###")
    print(f"### Reason: {result['reason']} ###")

    percentage = result.get("percentage", 0)
    print(f"### Investment Percentage: {percentage}% ###")

    order_executed = False
    trade_percentage = 0
    MIN_ORDER_KRW = 5100

    if result["decision"] == "buy":
        buy_amount_krw = my_krw * (percentage / 100)
        if buy_amount_krw > MIN_ORDER_KRW:
            print(f"### Buy Order: {buy_amount_krw:,.0f} KRW worth of {TRADE_COIN_SYMBOL} ###")
            try:
                bithumb.buy_market_order(f"KRW-{TRADE_COIN_SYMBOL}", buy_amount_krw)
                order_executed = True
                trade_percentage = percentage
            except Exception as e:
                print(f"### Buy Failed: {str(e)} ###")
        else:
            print(f"### Buy Failed: Amount ({buy_amount_krw:,.0f} KRW) below minimum ({MIN_ORDER_KRW} KRW) ###")

    elif result["decision"] == "sell":
        sell_coin_amount = my_coin * (percentage / 100)
        value = sell_coin_amount * current_coin_price

        if value > MIN_ORDER_KRW:
            print(f"### Sell Order: {sell_coin_amount:.8f} {TRADE_COIN_SYMBOL} ###")
            try:
                bithumb.sell_market_order(f"KRW-{TRADE_COIN_SYMBOL}", sell_coin_amount)
                order_executed = True
                trade_percentage = percentage
            except Exception as e:
                print(f"### Sell Failed: {str(e)} ###")
        else:
            print(f"### Sell Failed: Value ({value:,.0f} KRW) below minimum ({MIN_ORDER_KRW} KRW) ###")

    elif result["decision"] == "hold":
        print("### Hold Position ###")
        order_executed = True
        trade_percentage = 0

    time.sleep(1)

    updated_krw = bithumb.get_balance("KRW")
    updated_coin = bithumb.get_balance(TRADE_COIN_SYMBOL)
    updated_coin_price = get_current_price_from_bithumb_api(TRADE_COIN_SYMBOL)

    log_trade(
        conn,
        result["decision"],
        trade_percentage,
        result["reason"],
        TRADE_COIN_SYMBOL,
        updated_coin,
        updated_krw,
        updated_coin_price
    )

    conn.close()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 트레이딩 작업 완료")
    return {
        "decision": result["decision"],
        "percentage": trade_percentage,
        "reason": result["reason"],
        "updated_krw": updated_krw,
        "updated_coin": updated_coin,
        "updated_coin_price": updated_coin_price
    }


# --- PyQt6 GUI 관련 코드 ---

# API 키 설정을 위한 새 QDialog 클래스
class ApiSettingsDialog(QDialog):
    api_keys_saved = pyqtSignal() # API 키 저장 시 발생하는 시그널

    def __init__(self, current_api_keys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API 키 설정")
        self.setGeometry(200, 200, 500, 350) # 창 크기 조정
        self.current_api_keys = current_api_keys

        # 다이얼로그 전체에 스타일 적용
        self.setStyleSheet("background-color: #2d3748; color: #ecf0f1; font-size: 11pt;")
        self._init_ui()
        self.center_on_screen() # 화면 중앙에 배치

    def center_on_screen(self):
        screen_geometry = QApplication.primaryScreen().geometry()
        x = screen_geometry.center().x() - self.width() / 2
        y = screen_geometry.center().y() - self.height() / 2
        self.move(int(x), int(y))

    def _init_ui(self):
        layout = QVBoxLayout()

        form_layout = QGridLayout()
        form_layout.setSpacing(10) # 필드 간 간격

        # Bithumb Keys
        form_layout.addWidget(QLabel("Bithumb Access Key:"), 0, 0)
        self.bithumb_access_key_input = QLineEdit(self.current_api_keys.get("bithumb_access_key", ""))
        self.bithumb_access_key_input.setStyleSheet("background-color: #34495e; color: #ecf0f1; border: 1px solid #3498db; padding: 5px;")
        form_layout.addWidget(self.bithumb_access_key_input, 0, 1)

        form_layout.addWidget(QLabel("Bithumb Secret Key:"), 1, 0)
        self.bithumb_secret_key_input = QLineEdit(self.current_api_keys.get("bithumb_secret_key", ""))
        self.bithumb_secret_key_input.setEchoMode(QLineEdit.EchoMode.Password) # 비밀번호처럼 숨김
        self.bithumb_secret_key_input.setStyleSheet("background-color: #34495e; color: #ecf0f1; border: 1px solid #3498db; padding: 5px;")
        form_layout.addWidget(self.bithumb_secret_key_input, 1, 1)

        # SerpApi Key
        form_layout.addWidget(QLabel("SerpApi Key:"), 2, 0)
        self.serpapi_key_input = QLineEdit(self.current_api_keys.get("serpapi_api_key", ""))
        self.serpapi_key_input.setStyleSheet("background-color: #34495e; color: #ecf0f1; border: 1px solid #3498db; padding: 5px;")
        form_layout.addWidget(self.serpapi_key_input, 2, 1)

        # Gemini Key
        form_layout.addWidget(QLabel("Gemini API Key:"), 3, 0)
        self.gemini_api_key_input = QLineEdit(self.current_api_keys.get("gemini_api_key", ""))
        self.gemini_api_key_input.setEchoMode(QLineEdit.EchoMode.Password) # 비밀번호처럼 숨김
        self.gemini_api_key_input.setStyleSheet("background-color: #34495e; color: #ecf0f1; border: 1px solid #3498db; padding: 5px;")
        form_layout.addWidget(self.gemini_api_key_input, 3, 1)

        layout.addLayout(form_layout)
        layout.addSpacing(20) # 폼과 버튼 사이 간격

        warning_label = QLabel("경고: API 키는 민감한 정보입니다. 이 파일은 암호화되지 않습니다. "
                               "프로덕션 환경에서는 더 강력한 보안 조치를 고려하세요.")
        warning_label.setStyleSheet("color: #f56565; font-size: 9pt;") # 빨간색 경고
        warning_label.setWordWrap(True) # 텍스트 줄 바꿈
        layout.addWidget(warning_label)
        layout.addStretch(1) # 하단으로 밀어내기

        button_layout = QHBoxLayout()
        save_button = QPushButton("저장")
        save_button.setStyleSheet("font-size: 12pt; background-color: #4299e1; color: white; border-radius: 5px; padding: 8px 15px;")
        cancel_button = QPushButton("취소")
        cancel_button.setStyleSheet("font-size: 12pt; background-color: #e53e3e; color: white; border-radius: 5px; padding: 8px 15px;")
        save_button.clicked.connect(self.accept) # 저장 버튼 클릭 시 다이얼로그 닫고 QDialog.Accepted 반환
        cancel_button.clicked.connect(self.reject) # 취소 버튼 클릭 시 다이얼로그 닫고 QDialog.Rejected 반환
        button_layout.addStretch(1)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def accept(self):
        new_keys = {
            "bithumb_access_key": self.bithumb_access_key_input.text().strip(),
            "bithumb_secret_key": self.bithumb_secret_key_input.text().strip(),
            "serpapi_api_key": self.serpapi_key_input.text().strip(),
            "gemini_api_key": self.gemini_api_key_input.text().strip()
        }
        _save_api_keys(new_keys) # 파일에 저장
        self.api_keys_saved.emit() # 시그널 발생
        super().accept()


# 사용자 정의 AxisItem 클래스: 날짜/시간 포맷팅
class DateAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            dt = datetime.fromtimestamp(v)
            if spacing >= 3600 * 24:
                strings.append(dt.strftime('%m-%d'))
            elif spacing >= 3600:
                strings.append(dt.strftime('%m-%d %H:%M'))
            else:
                strings.append(dt.strftime('%H:%M:%S'))
        return strings

# 사용자 정의 AxisItem 클래스: 통화(KRW) 포맷팅
class CurrencyAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            strings.append(f"{v:,.0f}")
        return strings


# 백그라운드에서 트레이딩 로직을 실행할 Worker Thread
class TradeWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str) # 에러 메시지 시그널

    def run(self):
        try:
            trade_result = execute_trade_logic()
            self.finished.emit(trade_result)
        except Exception as e:
            error_msg = f"예상치 못한 오류 발생: {e}"
            self.error.emit(error_msg)
            print(f"TradeWorker Error: {e}")
            traceback.print_exc()


# 백그라운드에서 대시보드 데이터를 업데이트할 Worker Thread
class DashboardUpdater(QThread):
    data_updated = pyqtSignal(dict)
    error = pyqtSignal(str) # 에러 메시지 시그널

    def run(self):
        my_krw = 0.0
        my_coin = 0.0
        current_coin_price = 0.0
        news_articles = []
        recent_trades = []
        last_ai_decision = {"decision": "N/A", "percentage": "N/A", "reason": "No recent AI decision"}

        try:
            # API 키가 설정되어 있는지 확인
            if not BITHUMB_ACCESS_KEY or not BITHUMB_SECRET_KEY:
                raise ValueError("Bithumb API keys are not configured. Please set them in API Key Management.")

            bithumb = python_bithumb.Bithumb(BITHUMB_ACCESS_KEY, BITHUMB_SECRET_KEY)
            my_krw = bithumb.get_balance("KRW")
            my_coin = bithumb.get_balance(TRADE_COIN_SYMBOL)

            raw_current_coin_price = get_current_price_from_bithumb_api(TRADE_COIN_SYMBOL)
            print(f"DEBUG: Raw {TRADE_COIN_SYMBOL} price in DashboardUpdater: {raw_current_coin_price}")
            current_coin_price = raw_current_coin_price if raw_current_coin_price is not None else 0.0

        except Exception as e:
            error_message = f"Error fetching balance data: {e}"
            self.error.emit(error_message)
            print(error_message)
            traceback.print_exc()

        try:
            if SERPAPI_API_KEY:
                raw_news_articles = get_crypto_news_data(SERPAPI_API_KEY, TRADE_COIN_SYMBOL, "us", "en", 3)
                news_articles = [{"title": news_item['title'], "date": news_item['date']} for news_item in raw_news_articles]
            else:
                print("SerpApi API key is not set. Skipping news fetch.")
        except Exception as e:
            error_message = f"Error fetching news data: {e}"
            self.error.emit(error_message)
            print(error_message)
            traceback.print_exc()

        try:
            recent_trades = get_recent_trades_data(limit=5)
            if recent_trades:
                last_trade = recent_trades[0]
                last_ai_decision["decision"] = last_trade["decision"]
                last_ai_decision["percentage"] = last_trade["percentage"]
                last_ai_decision["reason"] = last_trade["reason"]
        except Exception as e:
            error_message = f"Error fetching recent trades: {e}"
            self.error.emit(error_message)
            print(error_message)
            traceback.print_exc()

        average_purchase_price = get_simple_average_purchase_price(TRADE_COIN_SYMBOL)
        profitability_rate_str = "N/A"
        if my_coin > 0:
            if average_purchase_price > 0:
                current_value_of_held_coin = my_coin * current_coin_price
                profitability_rate = ((current_value_of_held_coin - (my_coin * average_purchase_price)) / (
                            my_coin * average_purchase_price)) * 100
                profitability_rate_str = f"{profitability_rate:.2f}%"
            else:
                profitability_rate_str = "N/A (구매 내역 없음)"
        else:
            profitability_rate_str = "N/A (코인 미보유)"

        coin_current_market_value = 0.0
        if isinstance(my_coin, (int, float)) and isinstance(current_coin_price, (int, float)):
            coin_current_market_value = my_coin * current_coin_price
            total_value = my_krw + coin_current_market_value
        else:
            total_value = my_krw

        total_value_display_string = "N/A (데이터 오류)"
        if isinstance(my_krw, (int, float)) and isinstance(my_coin, (int, float)) and isinstance(current_coin_price, (int, float)):
            if current_coin_price > 0:
                total_value_calculated_numeric = my_krw + (my_coin * current_coin_price)
                total_value_display_string = f"{total_value_calculated_numeric:,.0f} KRW"
            elif my_coin == 0:
                total_value_calculated_numeric = my_krw
                total_value_display_string = f"{total_value_calculated_numeric:,.0f} KRW"
            else:
                total_value_display_string = f"{my_krw:,.0f} KRW + 보유 {TRADE_COIN_SYMBOL} 가치 알 수 없음"

        self.data_updated.emit({
            "current_balance": {
                "krw": my_krw,
                "coin_balance": my_coin,
                "coin_price": current_coin_price,
                "coin_current_market_value": coin_current_market_value,
                "total_value": total_value_display_string,
                "profitability_rate_str": profitability_rate_str
            },
            "news_articles": news_articles,
            "recent_trades": recent_trades,
            "last_ai_decision": last_ai_decision
        })


# --- CoinSettingsWindow 클래스 ---
class CoinSettingsWindow(QDialog):
    coin_symbol_saved = pyqtSignal(str)

    def __init__(self, current_trade_symbol, parent=None):
        super().__init__(parent)
        self.setWindowTitle("코인 설정")
        self.setGeometry(300, 300, 700, 500)
        self.current_trade_symbol = current_trade_symbol
        self.all_bithumb_coins = []

        self.setStyleSheet("background-color: #2d3748; color: #ecf0f1; font-size: 11pt;")

        self.init_ui()
        self.load_coins_and_set_initial_selection()
        self.center_on_screen()

    def center_on_screen(self):
        screen_geometry = QApplication.primaryScreen().geometry()
        x = screen_geometry.center().x() - self.width() / 2
        y = screen_geometry.center().y() - self.height() / 2
        self.move(int(x), int(y))

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<h2 style='color:#ecf0f1;'>빗썸 전체 코인</h2>"))

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("코인 검색...")
        self.search_input.setStyleSheet("background-color: #34495e; color: #ecf0f1; border: 1px solid #3498db; padding: 5px;")
        self.search_input.textChanged.connect(self.filter_bithumb_coins)
        left_layout.addWidget(self.search_input)

        self.available_coins_list = QListWidget()
        self.available_coins_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.available_coins_list.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #4a5568; color: #e2e8f0; border: 1px solid #2d3748;")
        left_layout.addWidget(self.available_coins_list)
        main_layout.addLayout(left_layout)

        middle_layout = QVBoxLayout()
        middle_layout.addStretch(1)

        self.add_button = QPushButton("추가 >>")
        self.add_button.setStyleSheet("font-size: 12pt; background-color: #89a9da; color: white; border-radius: 5px; padding: 8px 15px;")
        self.add_button.clicked.connect(self.add_coin_to_selected)
        middle_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("<< 삭제")
        self.remove_button.setStyleSheet("font-size: 12pt; background-color: #e1b673; color: white; border-radius: 5px; padding: 8px 15px;")
        self.remove_button.clicked.connect(self.remove_coin_from_selected)
        middle_layout.addWidget(self.remove_button)

        middle_layout.addStretch(1)

        self.save_button = QPushButton("저장")
        self.save_button.setStyleSheet("font-size: 13pt; font-weight: bold; color: #e2e8f0; background-color: #4299e1; border-radius: 5px; padding: 8px 15px;")
        self.save_button.clicked.connect(self.save_settings)
        middle_layout.addWidget(self.save_button)
        main_layout.addLayout(middle_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<h2 style='color:#ecf0f1;'>선택된 거래 코인</h2>"))
        self.selected_coins_list = QListWidget()
        self.selected_coins_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.selected_coins_list.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #4a5568; color: #e2e8f0; border: 1px solid #2d3748;")
        right_layout.addWidget(self.selected_coins_list)
        main_layout.addLayout(right_layout)

        main_layout.setStretch(0, 2)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 1)

        self.setStyleSheet("background-color: #2c3e50;")

    def fetch_bithumb_coins(self):
        """빗썸에서 현재 거래 가능한 모든 코인 심볼을 가져와서 available_coins_list에 채웁니다."""
        url = "https://api.bithumb.com/public/ticker/ALL_KRW"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "0000":
                coins = []
                for key in data.get("data", {}):
                    if key != "date":
                        coins.append(key)
                self.all_bithumb_coins = sorted(coins)
                self.populate_available_coins_list(self.all_bithumb_coins)
            else:
                QMessageBox.warning(self, "API 오류", f"빗썸 코인 목록을 가져오지 못했습니다. 상태 코드: {data.get('status')}")
        except requests.exceptions.RequestException as e:
            QMessageBox.critical(self, "네트워크 오류", f"빗썸 API 연결에 실패했습니다: {e}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"코인 목록 처리 중 예기치 않은 오류 발생: {e}")

    def populate_available_coins_list(self, coins_to_display):
        self.available_coins_list.clear()
        for coin in coins_to_display:
            self.available_coins_list.addItem(coin)

    def filter_bithumb_coins(self, text):
        filtered_coins = [coin for coin in self.all_bithumb_coins if text.lower() in coin.lower()]
        self.populate_available_coins_list(filtered_coins)

    def load_coins_and_set_initial_selection(self):
        """초기 코인 목록을 로드하고 현재 선택된 코인을 표시합니다."""
        self.fetch_bithumb_coins()

        if self.current_trade_symbol:
            self.selected_coins_list.clear()
            self.selected_coins_list.addItem(self.current_trade_symbol)

    def add_coin_to_selected(self):
        if self.selected_coins_list.count() >= 1:
            QMessageBox.warning(self, "경고", "하나의 코인만 선택하여 거래할 수 있습니다. 기존 코인을 먼저 삭제해주세요.")
            return

        selected_items = self.available_coins_list.selectedItems()
        if not selected_items:
            return

        coin_to_add = selected_items[0].text()

        for i in range(self.selected_coins_list.count()):
            if self.selected_coins_list.item(i).text() == coin_to_add:
                return

        self.selected_coins_list.addItem(coin_to_add)

    def remove_coin_from_selected(self):
        selected_items = self.selected_coins_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            self.selected_coins_list.takeItem(self.selected_coins_list.row(item))

    def save_settings(self):
        if self.selected_coins_list.count() == 0:
            QMessageBox.warning(self, "경고", "거래할 코인을 하나 이상 선택해야 합니다.")
            return
        selected_symbol = self.selected_coins_list.item(0).text()

        self.coin_symbol_saved.emit(selected_symbol)
        self.accept()


# --- TradingApp 클래스 (메인 윈도우) ---
class TradingApp(QMainWindow):
    TRADE_INTERVAL_SECONDS = 5 * 60  # 5분

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"JH [ {TRADE_COIN_SYMBOL} ] AI Trading Dashboard (Ver.2.2)")
        self.setGeometry(100, 100, 1200, 1100)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.init_ui()
        self.init_timers()
        init_db()

        self.auto_trade_running = False
        self.countdown_time = self.TRADE_INTERVAL_SECONDS

        self.update_dashboard_data()
        self.center_window()

    def center_window(self):
        screen = QApplication.primaryScreen()
        if screen:
            qr = self.frameGeometry()
            cp = screen.availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())
        else:
            print("Warning: Could not get primary screen information to center window.")

    def init_ui(self):
        self.main_layout.setSpacing(8)

        # 상단 타이틀 및 버튼 레이아웃
        top_bar_layout = QHBoxLayout()
        self.trade_dashboard_title_label = QLabel(f"JH [ {TRADE_COIN_SYMBOL} ] AI Trading Dashboard")
        self.trade_dashboard_title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.trade_dashboard_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # 중앙 정렬 유지
        self.trade_dashboard_title_label.setStyleSheet("color: #4299e1;")
        top_bar_layout.addWidget(self.trade_dashboard_title_label)

        # 타이틀과 버튼 사이에 공간을 추가하여 버튼을 오른쪽으로 밀어냅니다.
        top_bar_layout.addStretch(1)

        # API 키 관리 버튼 추가
        self.api_settings_btn = QPushButton("API 키 관리")
        self.api_settings_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.api_settings_btn.setStyleSheet(
            "background-color: #5a67d8; color: white; border-radius: 5px; padding: 6px 12px;")
        self.api_settings_btn.clicked.connect(self._open_api_settings_dialog)
        top_bar_layout.addWidget(self.api_settings_btn) # AlignRight 제거 (stretch가 이미 오른쪽으로 밀어냄)

        # 코인 설정 버튼 추가
        self.coin_settings_btn = QPushButton("코인 설정")
        self.coin_settings_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.coin_settings_btn.setStyleSheet(
            "background-color: #5a67d8; color: white; border-radius: 5px; padding: 6px 12px;")
        self.coin_settings_btn.clicked.connect(self._open_coin_settings_dialog)
        top_bar_layout.addWidget(self.coin_settings_btn) # AlignRight 제거

        self.main_layout.addLayout(top_bar_layout) # 이 레이아웃을 메인 레이아웃에 추가

        # 기존 top_buttons_layout 관련 코드 제거 (더 이상 필요 없음)
        # self.top_buttons_layout = QHBoxLayout()
        # self.top_buttons_layout.setContentsMargins(0, 0, 0, 10)
        # self.main_layout.addLayout(self.top_buttons_layout)

        self.resize(1200, 1000)

        status_controls_layout = QHBoxLayout()
        self.main_layout.addLayout(status_controls_layout)

        system_status_card = QWidget()
        system_status_card.setStyleSheet("background-color: #2d3748; border-radius: 10px; padding: 4px;")
        system_status_layout = QVBoxLayout(system_status_card)
        system_status_layout.setSpacing(2)
        system_status_layout.addWidget(
            QLabel("<h2 style='color:#63b3ed; font-size:18px; font-weight:bold;'>System Status</h2>"))

        dynamic_text_font = QFont("Arial", 11)

        self.current_time_label = QLabel("현재 시간: Loading...")
        self.current_time_label.setFont(dynamic_text_font)
        self.current_time_label.setStyleSheet("color: white;")

        self.trading_status_label = QLabel("트레이딩 상태: <span style='color:#ecc94b;'>정지됨</span>")
        self.trading_status_label.setFont(dynamic_text_font)
        self.trading_status_label.setStyleSheet("color: white;")

        self.countdown_label = QLabel("다음 거래까지: N/A")
        self.countdown_label.setFont(dynamic_text_font)
        self.countdown_label.setStyleSheet("color: white;")

        self.last_update_label = QLabel("마지막 업데이트: N/A")
        self.last_update_label.setFont(dynamic_text_font)
        self.last_update_label.setStyleSheet("color: white;")

        system_status_layout.addWidget(self.current_time_label)
        system_status_layout.addWidget(self.trading_status_label)
        system_status_layout.addWidget(self.countdown_label)
        system_status_layout.addWidget(self.last_update_label)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("자동 거래 시작")
        self.start_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.start_btn.setStyleSheet("background-color: #4299e1; color: white; border-radius: 5px; padding: 8px 15px;")
        self.start_btn.clicked.connect(self.start_auto_trade)
        self.stop_btn = QPushButton("자동 거래 중지")
        self.stop_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.stop_btn.setStyleSheet("background-color: #e53e3e; color: white; border-radius: 5px; padding: 8px 15px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_auto_trade)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        system_status_layout.addLayout(button_layout)
        status_controls_layout.addWidget(system_status_card)

        balance_card = QWidget()
        balance_card.setStyleSheet("background-color: #2d3748; border-radius: 10px; padding: 4px;")
        balance_layout = QVBoxLayout(balance_card)
        balance_layout.setSpacing(2)
        balance_layout.addWidget(
            QLabel("<h2 style='color:#63b3ed; font-size:18px; font-weight:bold;'>Current Balance</h2>"))

        self.krw_balance_label = QLabel("KRW: Loading...")
        self.krw_balance_label.setFont(dynamic_text_font)
        self.krw_balance_label.setStyleSheet("color: white;")
        balance_layout.addWidget(self.krw_balance_label)

        coin_info_layout = QHBoxLayout()
        coin_info_layout.setSpacing(10)

        self.coin_quantity_label = QLabel(f"보유 {TRADE_COIN_SYMBOL} (수량): Loading...")
        self.coin_quantity_label.setFont(dynamic_text_font)
        self.coin_quantity_label.setStyleSheet("color: white;")
        coin_info_layout.addWidget(self.coin_quantity_label)

        self.current_coin_price_inline_label = QLabel(f"{TRADE_COIN_SYMBOL} 현재가: Loading...")
        self.current_coin_price_inline_label.setFont(dynamic_text_font)
        self.current_coin_price_inline_label.setStyleSheet("color: white;")
        coin_info_layout.addWidget(self.current_coin_price_inline_label)
        coin_info_layout.addStretch(1)

        balance_layout.addLayout(coin_info_layout)

        self.coin_current_market_value_label = QLabel(f"{TRADE_COIN_SYMBOL} 현재 시장 가치: Loading...")
        self.coin_current_market_value_label.setFont(dynamic_text_font)
        self.coin_current_market_value_label.setStyleSheet("color: white;")

        self.total_value_label = QLabel("총 포트폴리오 가치 (KRW): Loading...")
        self.total_value_label.setFont(dynamic_text_font)
        self.total_value_label.setStyleSheet("color: white;")

        balance_layout.addWidget(self.coin_current_market_value_label)
        balance_layout.addWidget(self.total_value_label)
        status_controls_layout.addWidget(balance_card)

        ai_news_layout = QHBoxLayout()
        self.main_layout.addLayout(ai_news_layout)

        ai_decision_card = QWidget()
        ai_decision_card.setStyleSheet("background-color: #2d3748; border-radius: 10px; padding: 4px;")
        ai_decision_layout = QVBoxLayout(ai_decision_card)
        ai_decision_layout.setSpacing(2)
        ai_decision_layout.addWidget(
            QLabel("<h2 style='color:#63b3ed; font-size:18px; font-weight:bold;'>AI Decision</h2>"))

        decision_percentage_layout = QHBoxLayout()
        decision_percentage_layout.setSpacing(10)

        self.ai_decision_label = QLabel("결정: N/A")
        self.ai_decision_label.setFont(dynamic_text_font)
        self.ai_decision_label.setStyleSheet("color: white;")
        decision_percentage_layout.addWidget(self.ai_decision_label)

        self.ai_percentage_label = QLabel("비율: N/A")
        self.ai_percentage_label.setFont(dynamic_text_font)
        self.ai_percentage_label.setStyleSheet("color: white;")
        decision_percentage_layout.addWidget(self.ai_percentage_label)
        decision_percentage_layout.addStretch(1)

        ai_decision_layout.addLayout(decision_percentage_layout)

        self.ai_reason_list_widget = QListWidget()
        self.ai_reason_list_widget.setFont(dynamic_text_font)
        self.ai_reason_list_widget.setStyleSheet(
            "background-color: #4a5568; border: none; color: #e2e8f0; padding: 5px;")
        self.ai_reason_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.ai_reason_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ai_reason_list_widget.setWordWrap(True)

        ai_decision_layout.addWidget(self.ai_reason_list_widget)
        ai_news_layout.addWidget(ai_decision_card)

        news_card = QWidget()
        news_card.setStyleSheet("background-color: #2d3748; border-radius: 10px; padding: 4px;")
        news_layout = QVBoxLayout(news_card)
        news_layout.setSpacing(2)
        self.news_card_title_label = QLabel(
            f"<h2 style='color:#63b3ed; font-size:18px; font-weight:bold;'>Recent {TRADE_COIN_SYMBOL} News</h2>")
        news_layout.addWidget(self.news_card_title_label)
        self.news_list_widget = QListWidget()
        self.news_list_widget.setFont(dynamic_text_font)
        self.news_list_widget.setStyleSheet("background-color: #4a5568; border: none; color: #e2e8f0;")
        self.news_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.news_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.news_list_widget.setWordWrap(True)

        news_layout.addWidget(self.news_list_widget)
        ai_news_layout.addWidget(news_card)

        trades_card = QWidget()
        trades_card.setStyleSheet("background-color: #2d3748; border-radius: 10px; padding: 4px;")
        trades_layout = QVBoxLayout(trades_card)
        trades_layout.setSpacing(2)
        trades_layout.addWidget(
            QLabel("<h2 style='color:#63b3ed; font-size:18px; font-weight:bold;'>Recent Trades</h2>"))
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(7)
        self.trades_table.setHorizontalHeaderLabels([
            "타임스탬프", "결정", "비율", "이유", f"{TRADE_COIN_SYMBOL} 잔고", "KRW 잔고", f"{TRADE_COIN_SYMBOL} 가격"
        ])
        self.trades_table.horizontalHeader().setStyleSheet(
            "::section { background-color: #4a5568; color: #e2e8f0; font-size: 10pt; padding: 5px; }"
        )
        self.trades_table.setStyleSheet("background-color: #2d3748; color: #e2e8f0; gridline-color: #4a5568;")
        self.trades_table.verticalHeader().setVisible(False)
        self.trades_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.trades_table.horizontalHeader().setStretchLastSection(True)

        self.trades_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.trades_table.setFont(dynamic_text_font)
        self.trades_table.setWordWrap(True)
        trades_layout.addWidget(self.trades_table)
        self.main_layout.addWidget(trades_card)

        profit_graph_card = QWidget()
        profit_graph_card.setStyleSheet("background-color: #2d3748; border-radius: 10px; padding: 4px;")
        profit_graph_layout = QVBoxLayout(profit_graph_card)
        profit_graph_layout.setSpacing(2)
        profit_graph_layout.addWidget(
            QLabel("<h2 style='color:#63b3ed; font-size:18px; font-weight:bold;'>Profit Analysis</h2>"))

        self.profit_canvas = pg.PlotWidget(
            axisItems={'bottom': DateAxisItem(orientation='bottom'),
                       'left': CurrencyAxisItem(orientation='left')}
        )
        self.profit_canvas.setBackground('#2d3748')
        self.profit_canvas.getAxis('bottom').setTextPen('white')
        self.profit_canvas.getAxis('left').setTextPen('white')
        self.profit_canvas.getAxis('bottom').setPen('white')
        self.profit_canvas.getAxis('left').setPen('white')

        self.profit_canvas.getAxis('bottom').setHeight(30)

        profit_graph_layout.addWidget(self.profit_canvas)
        profit_graph_layout.addStretch(1)
        self.main_layout.addWidget(profit_graph_card)

        self.main_layout.setSpacing(8)
        status_controls_layout.setSpacing(20)
        ai_news_layout.setSpacing(20)

        status_controls_layout.setStretch(0, 1)
        status_controls_layout.setStretch(1, 1)

        ai_news_layout.setStretch(0, 1)
        ai_news_layout.setStretch(1, 1)

        # 레이아웃 스트레치 팩터 조정 (버튼 레이아웃이 상단 타이틀 레이아웃에 통합되었으므로 인덱스 조정)
        self.main_layout.setStretch(0, 0) # 상단 타이틀 및 버튼 레이아웃 (고정)
        self.main_layout.setStretch(1, 1) # System Status & Current Balance
        self.main_layout.setStretch(2, 3) # AI Decision & Recent News
        self.main_layout.setStretch(3, 3) # Recent Trades
        self.main_layout.setStretch(4, 5) # Profit Analysis


    def init_timers(self):
        self.current_time_timer = QTimer(self)
        self.current_time_timer.timeout.connect(self.update_time_display)
        self.current_time_timer.start(1000)

        self.dashboard_update_timer = QTimer(self)
        self.dashboard_update_timer.timeout.connect(self.update_dashboard_data)
        self.dashboard_update_timer.start(30 * 1000)

        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown_display)

    def update_time_display(self):
        now = datetime.now()
        self.current_time_label.setText(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        if not self.auto_trade_running:
            self.last_update_label.setText(f"마지막 업데이트: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    def update_countdown_display(self):
        if self.auto_trade_running:
            self.countdown_time -= 1
            minutes = self.countdown_time // 60
            seconds = self.countdown_time % 60
            self.countdown_label.setText(f"다음 거래까지: {minutes}분 {seconds}초")
            if self.countdown_time <= 0:
                self.countdown_time = self.TRADE_INTERVAL_SECONDS
                self.trigger_trade_cycle()

    def update_dashboard_data(self):
        self.dashboard_worker = DashboardUpdater()
        self.dashboard_worker.data_updated.connect(self.on_dashboard_data_updated)
        self.dashboard_worker.error.connect(self.on_worker_error)
        self.dashboard_worker.start()

    def on_dashboard_data_updated(self, data):
        print(f"DEBUG: update_gui_elements - Received data for GUI update: {data}")

        self.krw_balance_label.setText(
            f"KRW: {float(data['current_balance']['krw']):,.0f} KRW" if data['current_balance'][
                                                                            'krw'] is not None else "KRW: N/A")

        if data['current_balance']['coin_price'] is not None and isinstance(data['current_balance']['coin_price'],
                                                                            (int, float)):
            self.current_coin_price_inline_label.setText(
                f"{TRADE_COIN_SYMBOL} 현재가: {float(data['current_balance']['coin_price']):,.0f} KRW")
        else:
            self.current_coin_price_inline_label.setText(f"{TRADE_COIN_SYMBOL} 현재가: N/A")

        self.coin_quantity_label.setText(
            f"보유 {TRADE_COIN_SYMBOL} (수량): {float(data['current_balance']['coin_balance']):.4f} {TRADE_COIN_SYMBOL}" if
            data['current_balance']['coin_balance'] is not None else f"보유 {TRADE_COIN_SYMBOL} (수량): N/A")

        profitability_rate_str = data['current_balance'].get('profitability_rate_str', 'N/A')

        profit_color = "green"

        try:
            if profitability_rate_str != "N/A" and "N/A (" not in profitability_rate_str:
                profit_value = float(profitability_rate_str.replace('%', ''))
                if profit_value > 0:
                    profit_color = "#f56565"
                elif profit_value < 0:
                    profit_color = "#63b3ed"
        except ValueError:
            pass

        if data['current_balance']['coin_current_market_value'] is not None and isinstance(
                data['current_balance']['coin_current_market_value'], (int, float)):
            self.coin_current_market_value_label.setText(
                f"{TRADE_COIN_SYMBOL} 현재 시장 가치: {float(data['current_balance']['coin_current_market_value']):,.0f} KRW ( <span style='color:{profit_color};'>{profitability_rate_str}</span> )")
        else:
            self.coin_current_market_value_label.setText(
                f"{TRADE_COIN_SYMBOL} 현재 시장 가치: N/A (<span style='color:{profit_color};'>{profitability_rate_str}</span>)")

        self.total_value_label.setText(f"총 포트폴리오 가치 (KRW): {data['current_balance']['total_value']}")

        decision = data['last_ai_decision']['decision'].upper()
        percentage = data['last_ai_decision']['percentage']
        reason = data['last_ai_decision']['reason']
        self.ai_decision_label.setText(f"결정: {decision}")
        self.ai_percentage_label.setText(f"비율: {percentage}%")

        self.ai_reason_list_widget.clear()
        self.ai_reason_list_widget.addItem(f"이유: {reason}")

        if decision == 'BUY':
            self.ai_decision_label.setStyleSheet("color: #48bb78;")
        elif decision == 'SELL':
            self.ai_decision_label.setStyleSheet("color: #f56565;")
        else:
            self.ai_decision_label.setStyleSheet("color: #ecc94b;")

        self.news_list_widget.clear()
        if data['news_articles']:
            for news in data['news_articles']:
                item = QListWidgetItem(f"{news['title']} ({news['date']})")
                self.news_list_widget.addItem(item)
        else:
            self.news_list_widget.addItem("최근 뉴스 없음.")

        recent_trades = data.get('recent_trades', [])

        self.trades_table.setRowCount(0)

        self.trades_table.verticalHeader().setDefaultSectionSize(25)

        if recent_trades:
            self.trades_table.setRowCount(len(recent_trades))
            for row_idx, trade in enumerate(recent_trades):
                print(f"DEBUG: on_dashboard_data_updated - Populating row {row_idx} with trade data: {trade}")

                timestamp_str = trade.get('timestamp', 'N/A')
                if timestamp_str != 'N/A':
                    try:
                        timestamp_str = datetime.fromisoformat(timestamp_str).strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pass

                self.trades_table.setItem(row_idx, 0, QTableWidgetItem(timestamp_str))
                self.trades_table.setItem(row_idx, 1, QTableWidgetItem(trade.get('decision', 'N/A').upper()))
                self.trades_table.setItem(row_idx, 2, QTableWidgetItem(str(trade.get('percentage', 'N/A')) + '%'))

                reason_item = QTableWidgetItem(trade.get('reason', 'N/A'))
                self.trades_table.setItem(row_idx, 3, reason_item)

                self.trades_table.setItem(row_idx, 4, QTableWidgetItem(f"{trade.get('coin_balance', 0.0):.4f}"))
                self.trades_table.setItem(row_idx, 5, QTableWidgetItem(f"{trade.get('krw_balance', 0.0):,.0f}"))
                self.trades_table.setItem(row_idx, 6, QTableWidgetItem(f"{trade.get('coin_price', 0.0):,.0f}"))

            self.trades_table.resizeColumnsToContents()
            self.trades_table.setColumnWidth(3, 610)
        else:
            self.trades_table.setRowCount(1)
            self.trades_table.setItem(0, 0, QTableWidgetItem("최근 거래 없음."))
            self.trades_table.setSpan(0, 0, 1, 7)
            self.trades_table.item(0, 0).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self.update_profit_graph()

    def start_auto_trade(self):
        if self.auto_trade_running: return

        self.auto_trade_running = True
        self.trading_status_label.setText("트레이딩 상태: <span style='color:#48bb78;'>실행 중</span>")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.countdown_time = self.TRADE_INTERVAL_SECONDS
        self.countdown_timer.start(1000)

        self.trigger_trade_cycle()

    def stop_auto_trade(self):
        if not self.auto_trade_running: return

        self.auto_trade_running = False
        self.trading_status_label.setText("트레이딩 상태: <span style='color:#ecc94b;'>정지됨</span>")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.countdown_timer.stop()
        self.countdown_label.setText("다음 거래까지: N/A")
        self.countdown_time = self.TRADE_INTERVAL_SECONDS

    def trigger_trade_cycle(self):
        self.trade_worker = TradeWorker()
        self.trade_worker.finished.connect(self.on_trade_finished)
        self.trade_worker.error.connect(self.on_worker_error)

        self.start_btn.setText("거래 중...")
        self.start_btn.setEnabled(False)
        self.trade_worker.start()

    def on_trade_finished(self, trade_result):
        print("거래 사이클 완료:", trade_result)
        self.update_dashboard_data()

        if not self.auto_trade_running:
            self.start_btn.setText("자동 거래 시작")
            self.start_btn.setEnabled(True)
        else:
            self.start_btn.setText("자동 거래 실행 중...")

        self.last_update_label.setText(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def on_worker_error(self, message):
        QMessageBox.critical(self, "오류 발생", f"작업 중 오류가 발생했습니다: {message}")
        print(f"Worker Error: {message}")
        self.stop_auto_trade()
        self.start_btn.setText("자동 거래 시작")
        self.start_btn.setEnabled(True)

    def update_profit_graph(self):
        trades = get_all_trades_for_graph()  # 모든 거래 내역 가져오기
        if not trades:
            self.profit_canvas.clear()  # pyqtgraph는 axes.clear() 대신 plotwidget.clear()
            self.profit_canvas.setTitle('수익 분석을 표시할 거래 데이터가 없습니다.', color='white')
            # 텍스트를 중앙에 표시하는 더 좋은 방법 (PyQtGraph의 ViewBox를 직접 사용)
            self.profit_canvas.getPlotItem().hideAxis('bottom')
            self.profit_canvas.getPlotItem().hideAxis('left')
            return

        timestamps = []
        portfolio_values = []

        # 각 거래 시점의 총 포트폴리오 가치 계산
        for trade in trades:
            timestamp = datetime.fromisoformat(trade['timestamp'])
            krw_balance = trade['krw_balance']
            coin_balance = trade['coin_balance']
            coin_price = trade['coin_price']

            # None 값 처리 (데이터가 없을 경우 0으로 간주)
            if krw_balance is None: krw_balance = 0.0
            if coin_balance is None: coin_balance = 0.0
            if coin_price is None: coin_price = 0.0

            current_portfolio_value = krw_balance + (coin_balance * coin_price)
            timestamps.append(timestamp)
            portfolio_values.append(current_portfolio_value)

        # 첫 번째 포트폴리오 가치를 기준으로 누적 수익/손실 계산
        initial_value = portfolio_values[0] if portfolio_values else 0
        cumulative_pnl = [(value - initial_value) for value in portfolio_values]

        # PyQtGraph로 그래프 그리기
        self.profit_canvas.clear()  # 기존 그래프 지우기
        self.profit_canvas.getPlotItem().showAxis('bottom')  # 숨겼던 축 다시 표시
        self.profit_canvas.getPlotItem().showAxis('left')

        # x축 데이터를 타임스탬프 (초)로 변환하여 pyqtgraph가 처리하기 쉽게 함
        x_axis_data = [t.timestamp() for t in timestamps]

        # --- 점 제거: symbol, symbolSize, symbolBrush 인자 삭제 ---
        self.profit_canvas.plot(x=x_axis_data, y=cumulative_pnl,
                                pen=pg.mkPen(color='#48bb78', width=2)) # Tailwind green-500
        # --- 점 제거 끝 ---

        self.profit_canvas.setTitle('누적 수익/손실 변화', color='white', size='14pt')
        self.profit_canvas.setLabel('bottom', '시간', color='white')
        self.profit_canvas.setLabel('left', '수익/손실 (KRW)', color='white')  # 글자 색상 흰색으로 명시

        # 그리드 활성화 (PlotItem에서 그리드 가시성만 설정)
        self.profit_canvas.getPlotItem().showGrid(x=True, y=True)

        # --- 그리드 펜 색상 흰색으로 변경 및 투명도 조절 ---
        grid_pen = pg.mkPen(color=(255, 255, 255, int(0.3 * 255)))  # 흰색 (255,255,255) 및 투명도 30% (0.3 * 255)
        self.profit_canvas.getAxis('bottom').setPen(grid_pen)  # 축 선 색상 (옵션: 그리드와 같은 색상으로 맞춤)
        self.profit_canvas.getAxis('left').setPen(grid_pen)    # 축 선 색상 (옵션)
        self.profit_canvas.getAxis('bottom').setGrid(True)     # AxisItem에서 그리드 활성화
        self.profit_canvas.getAxis('left').setGrid(True)       # AxisItem에서 그리드 활성화
        # --- 그리드 펜 색상 변경 끝 ---

        self.profit_canvas.addLine(y=0, pen=pg.mkPen(color='red', width=0.8, style=Qt.PenStyle.DashLine))  # 0선 추가

        # X축과 Y축의 폰트 크기 조정
        self.profit_canvas.getAxis('bottom').setTickFont(QFont("Arial", 11))
        self.profit_canvas.getAxis('left').setTickFont(QFont("Arial", 11))

        # 자동 범위 조정
        self.profit_canvas.autoRange()

    def _open_api_settings_dialog(self):
        global BITHUMB_ACCESS_KEY, BITHUMB_SECRET_KEY, SERPAPI_API_KEY, GEMINI_API_KEY
        current_keys = {
            "bithumb_access_key": BITHUMB_ACCESS_KEY,
            "bithumb_secret_key": BITHUMB_SECRET_KEY,
            "serpapi_api_key": SERPAPI_API_KEY,
            "gemini_api_key": GEMINI_API_KEY
        }
        dialog = ApiSettingsDialog(current_keys, self)
        dialog.api_keys_saved.connect(self._on_api_keys_saved)
        dialog.exec()

    def _on_api_keys_saved(self):
        global BITHUMB_ACCESS_KEY, BITHUMB_SECRET_KEY, SERPAPI_API_KEY, GEMINI_API_KEY
        updated_keys = _load_api_keys()
        BITHUMB_ACCESS_KEY = updated_keys.get("bithumb_access_key", "")
        BITHUMB_SECRET_KEY = updated_keys.get("bithumb_secret_key", "")
        SERPAPI_API_KEY = updated_keys.get("serpapi_api_key", "")
        GEMINI_API_KEY = updated_keys.get("gemini_api_key", "")
        QMessageBox.information(self, "API 키 설정", "API 키가 성공적으로 저장되었습니다. "
                                "변경 사항을 완전히 적용하려면 애플리케이션을 재시작해야 합니다.")
        print("API keys updated and reloaded in application.")

    def _open_coin_settings_dialog(self):
        global TRADE_COIN_SYMBOL

        settings_window = CoinSettingsWindow(TRADE_COIN_SYMBOL, self)
        settings_window.coin_symbol_saved.connect(self._on_coin_symbol_saved)
        settings_window.exec()

    def _on_coin_symbol_saved(self, new_symbol):
        global TRADE_COIN_SYMBOL

        if new_symbol != TRADE_COIN_SYMBOL:
            TRADE_COIN_SYMBOL = new_symbol
            _save_trade_settings(new_symbol)

            QMessageBox.information(self, "코인 설정",
                                    f"거래 코인이 {new_symbol}으로 변경되었습니다. 변경 사항을 완전히 적용하려면 "
                                    f"애플리케이션을 재시작해야 합니다.")

            self.stop_auto_trade()

            self.setWindowTitle(f"JH [ {TRADE_COIN_SYMBOL} ] AI Trading Dashboard (Ver.2.2)")
            self.trade_dashboard_title_label.setText(f"JH [ {TRADE_COIN_SYMBOL} ] AI Trading Dashboard")

            self.coin_quantity_label.setText(f"보유 {TRADE_COIN_SYMBOL} (수량): Loading...")
            self.current_coin_price_inline_label.setText(f"{TRADE_COIN_SYMBOL} 현재가: Loading...")
            self.coin_current_market_value_label.setText(f"{TRADE_COIN_SYMBOL} 현재 시장 가치: Loading...")

            self.news_card_title_label.setText(
                f"<h2 style='color:#63b3ed; font-size:18px; font-weight:bold;'>Recent {TRADE_COIN_SYMBOL} News</h2>")

            self.trades_table.setHorizontalHeaderLabels([
                "타임스탬프", "결정", "비율", "이유", f"{TRADE_COIN_SYMBOL} 잔고", "KRW 잔고", f"{TRADE_COIN_SYMBOL} 가격"
            ])

            self.update_dashboard_data()


# 처리되지 않은 예외를 잡기 위한 후크
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    msg = f"예기치 않은 오류 발생: {exc_type.__name__}: {exc_value}\\n"
    msg += "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(msg)
    QMessageBox.critical(None, "예기치 않은 오류", "자동 트레이딩 시스템에 예기치 않은 오류가 발생하여 종료됩니다.\n자세한 내용은 콘솔을 확인하세요.")
    sys.exit(1)


if __name__ == "__main__":
    sys.excepthook = handle_exception

    app = QApplication(sys.argv)
    window = TradingApp()
    window.show()
    window.start_auto_trade()  # 이 줄을 추가하여 프로그램 시작 시 자동 거래를 시작합니다.
    sys.exit(app.exec())
