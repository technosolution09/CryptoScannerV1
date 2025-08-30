#!/usr/bin/env python3
"""
AI Momentum Crypto Trading Bot - Delta Exchange (via CCXT)
--------------------------------------------------------
Scans top derivative/spot pairs on Delta Exchange, scores momentum using
RSI / MACD / QQE / Volume and optionally places trades via CCXT.
Defaults to dry-run. Carefully review and test before using with real keys.

Requirements:
  pip install ccxt pandas numpy ta

Notes for Delta Exchange:
 - CCXT includes a 'delta' exchange class. Delta often lists derivative contracts.
 - Markets may be futures/perpetuals; symbol formats may vary. We filter markets
   by quote currency (USDT/USD) and 'active' flag. Adjust filters as needed.
"""

import time
import math
import json
import traceback
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt  # pip install ccxt
except Exception as e:
    ccxt = None

try:
    import ta    # pip install ta
except Exception as e:
    ta = None


@dataclass
class BotConfig:
    exchange_id: str = "delta"      # Delta Exchange via CCXT
    api_key: str = ""               # set your Delta API key here
    api_secret: str = ""            # set your Delta API secret here
    password: str = ""

    base_quote: str = "USDT"        # target quote to filter markets by (USDT/USD)
    timeframe: str = "1h"
    lookback_candles: int = 600
    top_n_by_volume: int = 50

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_fast: int = 50
    ema_slow: int = 200
    volume_ma: int = 20

    qqe_rsi_period: int = 14
    qqe_smooth: int = 5
    qqe_factor: float = 4.236

    min_score_to_buy: float = 3.0
    min_score_to_sell: float = 3.0

    risk_per_trade: float = 0.005
    atr_period: int = 14
    atr_mult_sl: float = 2.0
    atr_mult_tp: float = 3.0
    max_concurrent_positions: int = 5
    min_notional_usd: float = 10.0

    dry_run: bool = True
    poll_interval_sec: int = 300


# ---------------- indicator helpers ----------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    return rma(true_range(df), period)

def qqe(df: pd.DataFrame, rsi_period: int = 14, smooth: int = 5, factor: float = 4.236) -> Tuple[pd.Series, pd.Series]:
    rsi = ta.momentum.rsi(df["close"], window=rsi_period, fillna=False)
    rsi_smooth = rma(rsi, smooth)
    rsi_delta = (rsi_smooth - rsi_smooth.shift(1)).abs()
    wilders = rma(rsi_delta, rsi_period)
    long = pd.Series(index=df.index, dtype=float)
    short = pd.Series(index=df.index, dtype=float)
    long.iloc[:] = np.nan
    short.iloc[:] = np.nan
    # initialize if possible
    valid_idx = rsi_smooth.first_valid_index()
    if valid_idx is None:
        return long, rsi_smooth
    init_i = rsi_smooth.index.get_loc(valid_idx)
    long.iloc[init_i] = rsi_smooth.iloc[init_i]
    short.iloc[init_i] = rsi_smooth.iloc[init_i]
    for i in range(init_i+1, len(df)):
        if np.isnan(rsi_smooth.iloc[i]) or np.isnan(wilders.iloc[i]):
            continue
        step = factor * wilders.iloc[i]
        prev_long = long.iloc[i-1] if not np.isnan(long.iloc[i-1]) else rsi_smooth.iloc[i-1]
        if rsi_smooth.iloc[i] > prev_long:
            candidate_long = max(prev_long, rsi_smooth.iloc[i] - step)
        else:
            candidate_long = rsi_smooth.iloc[i] - step
        long.iloc[i] = candidate_long
        prev_short = short.iloc[i-1] if not np.isnan(short.iloc[i-1]) else rsi_smooth.iloc[i-1]
        if rsi_smooth.iloc[i] < prev_short:
            candidate_short = min(prev_short, rsi_smooth.iloc[i] + step)
        else:
            candidate_short = rsi_smooth.iloc[i] + step
        short.iloc[i] = candidate_short
    return long, rsi_smooth

def macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def indicator_pack(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg.ema_fast)
    out["ema_slow"] = ema(out["close"], cfg.ema_slow)
    out["rsi"] = ta.momentum.rsi(out["close"], window=cfg.rsi_period, fillna=False)
    macd_line, signal_line, hist = macd(out["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["macd_line"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    qqe_line, rsi_s = qqe(out, cfg.qqe_rsi_period, cfg.qqe_smooth, cfg.qqe_factor)
    out["qqe"] = qqe_line
    out["rsi_smooth"] = rsi_s
    out["vol_ma"] = out["volume"].rolling(cfg.volume_ma).mean()
    out["atr"] = atr(out, cfg.atr_period)
    return out

def momentum_score(row_prev: pd.Series, row: pd.Series) -> float:
    score = 0.0
    if row["close"] > row["ema_fast"]: score += 1
    if row["close"] > row["ema_slow"]: score += 1
    if row["ema_fast"] > row["ema_slow"]: score += 1
    if row["rsi"] > 50: score += 1
    if row_prev["rsi"] < row["rsi"]: score += 1
    if row["macd_hist"] > 0: score += 1
    if row["macd_hist"] > row_prev["macd_hist"]: score += 1
    if row["rsi_smooth"] > row["qqe"]: score += 1
    if row["qqe"] > row_prev["qqe"]: score += 1
    if row["volume"] > row["vol_ma"]: score += 1
    return float(score)

def sell_pressure_score(row_prev: pd.Series, row: pd.Series) -> float:
    score = 0.0
    if row["close"] < row["ema_fast"]: score += 1
    if row["close"] < row["ema_slow"]: score += 1
    if row["ema_fast"] < row["ema_slow"]: score += 1
    if row["rsi"] < 50: score += 1
    if row_prev["rsi"] > row["rsi"]: score += 1
    if row["macd_hist"] < 0: score += 1
    if row["macd_hist"] < row_prev["macd_hist"]: score += 1
    if row["rsi_smooth"] < row["qqe"]: score += 1
    if row["qqe"] < row_prev["qqe"]: score += 1
    if row["volume"] > row["vol_ma"]: score += 1
    return float(score)


# ---------------- Exchange wrapper for Delta ----------------
class Exchange:
    def __init__(self, cfg: BotConfig):
        if ccxt is None:
            raise RuntimeError("ccxt is not installed. pip install ccxt")
        ex_class = getattr(ccxt, cfg.exchange_id)
        self.ex = ex_class({
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "password": cfg.password or None,
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True}
        })
        self.cfg = cfg
        self.ex.load_markets()

    def top_symbols_by_volume(self, limit: int) -> List[str]:
        tickers = self.ex.fetch_tickers()
        rows = []
        for sym, t in tickers.items():
            # Normalize: use only markets with desired quote (USDT/USD) and active
            market = self.ex.markets.get(sym, None)
            if not market:
                continue
            quote = market.get("quote") or market.get("quoteId") or ""
            active = market.get("active", True)
            if not active:
                continue
            if quote.upper() not in (self.cfg.base_quote.upper(), "USD"):
                continue
            qv = t.get("quoteVolume") or t.get("baseVolume") or 0
            rows.append((sym, qv))
        rows.sort(key=lambda x: x[1] or 0, reverse=True)
        return [r[0] for r in rows[:limit]]

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ohlcv = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def balance_usd(self) -> float:
        bal = self.ex.fetch_balance()
        total = bal.get("total", {})
        usd = float(total.get(self.cfg.base_quote, 0))
        return usd

    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        return self.ex.create_order(symbol, "market", side, amount)

    def create_oco(self, symbol: str, side: str, amount: float, stop_price: float, limit_price: float, take_profit: float) -> None:
        try:
            # many derivative exchanges do not support simple OCO via unified CCXT;
            # placeholder: create stop market (if supported) + limit TP
            params = {}
            self.ex.create_order(symbol, "stop_market", "sell" if side == "buy" else "buy", amount, None, {"stopPrice": stop_price})
            self.ex.create_order(symbol, "limit", "sell" if side == "buy" else "buy", amount, take_profit)
        except Exception as e:
            print(f"[WARN] OCO/Bracket not supported for {symbol}: {e}")


# ---------------- Bot core ----------------
class MomentumBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.ex = Exchange(cfg)

    def universe(self) -> List[str]:
        try:
            return self.ex.top_symbols_by_volume(self.cfg.top_n_by_volume)
        except Exception as e:
            print("[WARN] top_symbols_by_volume failed:", e)
            # fallback
            return list(self.ex.ex.markets.keys())[: self.cfg.top_n_by_volume]

    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            df = self.ex.fetch_ohlcv_df(symbol, self.cfg.timeframe, self.cfg.lookback_candles)
            if df.shape[0] < 100:
                return None
            feat = indicator_pack(df, self.cfg).dropna()
            if feat.empty or feat.shape[0] < 5:
                return None
            row = feat.iloc[-1]
            row_prev = feat.iloc[-2]
            buy_score = momentum_score(row_prev, row)
            sell_score = sell_pressure_score(row_prev, row)
            action = "hold"
            if buy_score >= self.cfg.min_score_to_buy and buy_score >= sell_score + 1:
                action = "buy"
            elif sell_score >= self.cfg.min_score_to_sell and sell_score >= buy_score + 1:
                action = "sell"
            stop_distance = row["atr"] * self.cfg.atr_mult_sl
            take_distance = row["atr"] * self.cfg.atr_mult_tp
            last = row["close"]
            sl = max(0.00000001, last - stop_distance) if action == "buy" else last + stop_distance
            tp = last + take_distance if action == "buy" else last - take_distance
            return {
                "symbol": symbol,
                "price": float(last),
                "buy_score": buy_score,
                "sell_score": sell_score,
                "action": action,
                "sl": float(sl),
                "tp": float(tp),
                "atr": float(row["atr"]),
                "volume": float(row["volume"]),
                "vol_ma": float(row["vol_ma"]),
            }
        except Exception as e:
            print(f"[ERR] analyze_symbol {symbol}: {e}")
            traceback.print_exc()
            return None

    def position_size(self, equity_usd: float, price: float, atr_value: float) -> float:
        risk_usd = equity_usd * self.cfg.risk_per_trade
        denom = max(1e-9, atr_value * self.cfg.atr_mult_sl)
        qty = risk_usd / denom
        amount = max(0.0, qty)
        if amount * price < self.cfg.min_notional_usd:
            amount = self.cfg.min_notional_usd / price
        return float(amount)

    def maybe_execute(self, idea: Dict) -> None:
        action = idea["action"]
        if action not in ("buy", "sell"):
            return
        if self.cfg.dry_run:
            print(f"[DRY] {action.upper()} {idea['symbol']} price={idea['price']:.4f} SL={idea['sl']:.4f} TP={idea['tp']:.4f}  (scores: buy={idea['buy_score']:.1f}, sell={idea['sell_score']:.1f})")
            return
        try:
            equity = self.ex.balance_usd()
            amount = self.position_size(equity, idea["price"], idea["atr"])
            amount = float("{:.6f}".format(amount))
            side = "buy" if action == "buy" else "sell"
            print(f"[LIVE] Placing {side} {idea['symbol']} amount={amount} price~{idea['price']:.4f}")
            order = self.ex.create_market_order(idea['symbol'], side, amount)
            print("[LIVE] Market order placed:", order)
            self.ex.create_oco(idea['symbol'], side, amount, idea['sl'], idea['sl'], idea['tp'])
        except Exception as e:
            print(f"[ERR] execution failed for {idea['symbol']}: {e}")
            traceback.print_exc()

    def run_once(self) -> List[Dict]:
        syms = self.universe()
        print(f"[INFO] Scanning {len(syms)} symbols...")
        ideas: List[Dict] = []
        for s in syms:
            idea = self.analyze_symbol(s)
            if idea is not None:
                ideas.append(idea)
        ideas.sort(key=lambda x: max(x["buy_score"], x["sell_score"]), reverse=True)
        top = ideas[:10]
        print(json.dumps(top, indent=2))
        open_actions = [i for i in ideas if i["action"] in ("buy", "sell")]
        for idea in open_actions[: self.cfg.max_concurrent_positions]:
            self.maybe_execute(idea)
        return ideas

    def run_loop(self):
        while True:
            try:
                self.run_once()
            except Exception as e:
                print("[FATAL]", e)
                traceback.print_exc()
            time.sleep(self.cfg.poll_interval_sec)


def main():
    cfg = BotConfig(
        exchange_id="delta",
        api_key="",        # <-- paste your delta API key
        api_secret="",     # <-- paste your delta API secret
        base_quote="USDT",
        timeframe="1h",
        lookback_candles=600,
        dry_run=True,
        poll_interval_sec=300
    )
    if ta is None:
        raise RuntimeError("The 'ta' package is required. Run: pip install ta")
    bot = MomentumBot(cfg)
    bot.run_once()


if __name__ == "__main__":
    main()
