import os, datetime as dt
import numpy as np, pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
import requests
import openai

# ---------- Config ----------
TICKERS = os.getenv("TICKERS", "RELIANCE.NS,TCS.NS,HDFCBANK.NS,ICICIBANK.NS,NIFTYBEES.NS").split(",")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "120"))
MA = int(os.getenv("MA_WINDOW", "20"))
RSI_WIN = int(os.getenv("RSI_WINDOW", "14"))
RSI_BUY = int(os.getenv("RSI_BUY", "55"))
RSI_SELL = int(os.getenv("RSI_SELL", "45"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

def fetch_df(ticker):
    end = dt.date.today()
    start = end - dt.timedelta(days=LOOKBACK_DAYS)
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    if df.empty: return None
    df = df.rename(columns=str.lower)
    df["ma"] = df["close"].rolling(MA).mean()
    df["rsi"] = RSIIndicator(df["close"], window=RSI_WIN).rsi()
    df = df.dropna()
    return df

def classify(row):
    above_ma = row["close"] > row["ma"]
    if above_ma and row["rsi"] >= RSI_BUY: return "Bullish"
    if (not above_ma) and row["rsi"] <= RSI_SELL: return "Bearish"
    return "Neutral"

def make_plaintext(rows):
    # No LLM fallback: readable summary
    lines = [f"• {t}: {lab} (Close {c:.2f}, MA{MA} {m:.2f}, RSI {r:.0f})" for t,lab,c,m,r in rows]
    tip = "Tip: Momentum above MA with healthy RSI can indicate strength. This is educational, not advice."
    return "📊 AI Market Pulse — " + dt.date.today().isoformat() + "\n\n" + "\n".join(lines) + "\n\n" + tip

def summarize_with_llm(rows):
    if not OPENAI_API_KEY:  # fall back if no key set
        return make_plaintext(rows)
    bullets = "\n".join([f"- {t}: {lab} (close {c:.2f}, MA{MA} {m:.2f}, RSI {r:.0f})" for t,lab,c,m,r in rows])
    prompt = f"""You are a finance educator for Indian markets. Convert the data into a short, neutral,
educational Telegram post. Title + 4–6 bullets + 1-line tip. Avoid advice/targets.

Data:
{bullets}
"""
    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    title_and_bullets = res.choices[0].message["content"].strip()
    return f"{title_and_bullets}\n\n⚠️ Educational only. Not investment advice."

def post_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True})

def main():
    rows = []
    for t in TICKERS:
        df = fetch_df(t)
        if df is None or df.empty: continue
        last = df.iloc[-1]
        label = classify(last)
        rows.append((t, label, float(last["close"]), float(last["ma"]), float(last["rsi"])))
    if not rows:
        post_telegram("No data today.")
        return
    msg = summarize_with_llm(rows)
    post_telegram(msg)

if __name__ == "__main__":
    main()
