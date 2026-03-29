"""
============================================================
  MINERVINI SEPA STRATEGY — 15-YEAR PYTHON BACKTESTER
  Specific Entry Point Analysis (Stage 2 + VCP + RS Rating)
  Backtest Period: 2010 – 2024

  v3 CHANGES vs v2:
    1. Market filter: only enter new trades when SPY > its 200MA
       (Minervini goes to cash in bear markets — avoids 2018/2020/2022)
    2. Breakout confirmation: require 2 consecutive closes above pivot
       (reduces false breakouts / getting stopped out immediately)
    3. Volume relaxed back to 1.75x (balance quality vs quantity)
    4. Stop loss widened slightly to 8% (less noise-driven stops)
    5. Yearly ROI % now printed in the report (not just Net $)
============================================================

INSTALL:
    pip install yfinance pandas numpy matplotlib tqdm

RUN:
    python minervini_sepa_backtest.py

OUTPUTS:
    - Console: full stats + year-by-year ROI% + top trades
    - minervini_results.png   (6-panel chart)
    - minervini_trades.csv    (every trade, opens in Excel)

ENTRY LOGIC (all must be TRUE):
    ── Market Filter (new in v3) ────────────────────────────
    0. SPY Close > SPY MA200  (only trade in uptrending markets)

    ── Trend Template ──────────────────────────────────────
    1. Close > MA50 > MA150 > MA200
    2. Close >= 30% above 52-week low
    3. Close within 25% of 52-week high
    4. MA200 is rising (higher than 20 trading days ago)

    ── Relative Strength ───────────────────────────────────
    5. Stock 12-month return > SPY 12-month return

    ── VCP (Volatility Contraction Pattern) ────────────────
    6. 10-day ATR / price < 3.5%
    7. 10-day avg volume < 50-day avg volume

    ── Breakout + Confirmation (new in v3) ─────────────────
    8. Close > highest close of prior 20 sessions
    9. This is the 2nd consecutive day above the pivot  ← new
   10. Volume >= 1.75x the 50-day avg volume

EXIT LOGIC (first to trigger wins):
    A. Hard Stop:      -8% from entry
    B. Trailing Stop:  close < MA50 after being +5% profitable
    C. Time Stop:      120 days max hold
============================================================
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
CONFIG = {
    "start_date": "2010-01-01",
    "end_date":   "2024-12-31",

    "tickers": [
        # Mega-cap tech & semis
        "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "QCOM", "AMAT", "LRCX", "KLAC", "MRVL",
        # Internet / platforms
        "GOOGL", "META", "AMZN", "NFLX", "UBER", "SNAP", "PINS",
        # Enterprise software
        "CRM", "NOW", "ADBE", "INTU", "WDAY", "VEEV", "PAYC", "PCTY", "COUP",
        # Cybersecurity / cloud
        "CRWD", "PANW", "ZS", "NET", "DDOG", "SNOW", "OKTA", "TWLO", "ESTC",
        # Ad tech / marketplaces
        "TTD", "BILL", "HUBS", "MELI", "SE", "SHOP",
        # Consumer discretionary
        "LULU", "DECK", "CELH", "ELF", "MNST", "POOL", "RH", "ONON", "CROX",
        # Healthcare & biotech
        "LLY", "NVO", "ISRG", "REGN", "ALGN", "DXCM", "IDXX", "ABMD",
        "MEDP", "GMED", "TMDX", "MRNA", "PCVX", "RXRX",
        # Industrials / specialty
        "ODFL", "BLDR", "MLI", "HWM", "FN", "BWXT", "AXON", "FICO",
        "CASY", "TREX", "KTOS", "RKLB", "SAIA",
        # Financials / fintech
        "V", "MA", "PYPL", "COIN", "AFRM", "SQ",
        # Energy / materials
        "CEIX", "AMR", "BTU",
        # Additional growth
        "ENPH", "FSLR", "GNRC", "LSTR", "INSP", "DRVN",
    ],

    # ── Market filter ─────────────────────────────────────────────
    "market_filter":      True,   # ← NEW: only trade when SPY > SPY MA200

    # ── Trend Template ───────────────────────────────────────────
    "pct_above_52w_low":  0.30,
    "pct_below_52w_high": 0.25,
    "ma200_rising_days":  20,

    # ── Relative Strength ────────────────────────────────────────
    "rs_vs_spy":          True,

    # ── VCP ──────────────────────────────────────────────────────
    "vcp_atr_threshold":   0.035,
    "vcp_vol_contraction": True,

    # ── Breakout ─────────────────────────────────────────────────
    "breakout_lookback":      20,
    "breakout_vol_mult":      1.75,  # ← relaxed from 2.0x
    "breakout_confirm_days":  2,     # ← NEW: 2 consecutive closes above pivot

    # ── Exits ────────────────────────────────────────────────────
    "stop_loss_pct":      0.08,   # ← widened slightly from 7%
    "profit_target_pct":  9.99,   # disabled — trailing stop handles exits
    "max_hold_days":      120,
    "trend_break_exit":   True,   # exit if close < MA50 when up >= 5%

    # ── Portfolio ────────────────────────────────────────────────
    "max_positions":      6,
    "position_size_pct":  0.15,
    "starting_capital":   100_000,

    # ── Output ───────────────────────────────────────────────────
    "chart_file":  "minervini_results.png",
    "trades_file": "minervini_trades.csv",
}


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — DOWNLOAD
# ══════════════════════════════════════════════════════════════════

def download_all(cfg):
    tickers = list(set(cfg["tickers"] + ["SPY"]))
    buffer_start = (
        datetime.strptime(cfg["start_date"], "%Y-%m-%d") - timedelta(days=310)
    ).strftime("%Y-%m-%d")

    print(f"\n{'═'*62}")
    print(f"  MINERVINI SEPA v3  |  15-YEAR BACKTEST  |  2010–2024")
    print(f"{'═'*62}")
    print(f"\n📥 Downloading {len(tickers)} tickers ({buffer_start} → {cfg['end_date']})...")

    raw = yf.download(
        tickers,
        start=buffer_start,
        end=cfg["end_date"],
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    data = {}
    for t in tickers:
        try:
            df = raw.xs(t, axis=1, level=1).dropna()
            if len(df) >= 252:
                data[t] = df
        except Exception:
            continue

    print(f"✅ Loaded {len(data)} tickers.\n")
    return data


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — INDICATORS
# ══════════════════════════════════════════════════════════════════

def build_indicators(df, rising_days=20):
    df = df.copy()
    c, v, h, l = df["Close"], df["Volume"], df["High"], df["Low"]

    df["ma50"]      = c.rolling(50).mean()
    df["ma150"]     = c.rolling(150).mean()
    df["ma200"]     = c.rolling(200).mean()
    df["ma200_lag"] = df["ma200"].shift(rising_days)
    df["high_52w"]  = c.rolling(252).max()
    df["low_52w"]   = c.rolling(252).min()

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr10"] = tr.rolling(10).mean()

    df["vol10"]  = v.rolling(10).mean()
    df["vol50"]  = v.rolling(50).mean()

    # Pivot: highest close of prior N days (shifted so today not included)
    df["pivot"]  = c.shift(1).rolling(20).max()

    # Days consecutively above pivot (for confirmation filter)
    above_pivot  = (c > df["pivot"]).astype(int)
    df["consec_above_pivot"] = above_pivot.groupby(
        (above_pivot != above_pivot.shift()).cumsum()
    ).cumcount() + 1
    df.loc[above_pivot == 0, "consec_above_pivot"] = 0

    df["ret12m"] = c.pct_change(252)

    return df.dropna()


# ══════════════════════════════════════════════════════════════════
#  STEP 3 — ENTRY SIGNAL
# ══════════════════════════════════════════════════════════════════

def is_entry(row, spy_ret12m, spy_above_200, cfg):
    c = row["Close"]

    # 0. Market filter: SPY must be above its 200MA
    if cfg["market_filter"] and not spy_above_200:
        return False

    # 1. Trend stack
    if not (c > row["ma50"] > row["ma150"] > row["ma200"]):
        return False
    # 2. >= 30% above 52-week low
    if c < row["low_52w"] * (1 + cfg["pct_above_52w_low"]):
        return False
    # 3. Within 25% of 52-week high
    if c < row["high_52w"] * (1 - cfg["pct_below_52w_high"]):
        return False
    # 4. MA200 rising
    if pd.isna(row["ma200_lag"]) or row["ma200"] <= row["ma200_lag"]:
        return False
    # 5. RS vs SPY
    if cfg["rs_vs_spy"] and (pd.isna(row["ret12m"]) or row["ret12m"] <= spy_ret12m):
        return False
    # 6. VCP tightness
    if (row["atr10"] / c) >= cfg["vcp_atr_threshold"]:
        return False
    # 7. Volume drying up in base
    if cfg["vcp_vol_contraction"] and row["vol10"] >= row["vol50"]:
        return False
    # 8. Above pivot
    if pd.isna(row["pivot"]) or c <= row["pivot"]:
        return False
    # 9. Confirmation: N consecutive days above pivot
    if row["consec_above_pivot"] < cfg["breakout_confirm_days"]:
        return False
    # 10. Volume surge
    if row["Volume"] < row["vol50"] * cfg["breakout_vol_mult"]:
        return False

    return True


# ══════════════════════════════════════════════════════════════════
#  STEP 4 — BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

def run_backtest(data, cfg):
    start_dt  = pd.Timestamp(cfg["start_date"])
    equity    = float(cfg["starting_capital"])
    positions = {}
    trades    = []
    eq_curve  = []

    spy_df  = build_indicators(data["SPY"], cfg["ma200_rising_days"])
    spy_ret = spy_df["ret12m"]

    print("⚙️  Building indicators...")
    inds = {}
    for t, df in tqdm(data.items(), desc="Indicators", ncols=70):
        try:
            inds[t] = build_indicators(df, cfg["ma200_rising_days"])
        except Exception:
            pass

    all_dates = spy_df[spy_df.index >= start_dt].index
    print(f"\n🔁 Simulating {len(all_dates):,} trading days across {len(inds)} tickers...\n")

    for date in tqdm(all_dates, desc="Backtesting", ncols=70):

        # SPY state for this day
        spy_row        = spy_df.loc[date] if date in spy_df.index else None
        spy_above_200  = bool(spy_row is not None and
                              float(spy_row["Close"]) > float(spy_row["ma200"]))
        spy_r          = float(spy_ret.get(date, np.nan)) if date in spy_ret.index else np.nan

        # ── EXITS ─────────────────────────────────────────────────
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            idf = inds.get(ticker)
            if idf is None or date not in idf.index:
                continue

            row   = idf.loc[date]
            price = float(row["Close"])
            ret   = (price - pos["entry_price"]) / pos["entry_price"]
            held  = (date - pos["entry_date"]).days
            reason = None

            if ret <= -cfg["stop_loss_pct"]:
                reason = "STOP"
            elif ret >= cfg["profit_target_pct"]:
                reason = "TARGET"
            elif held >= cfg["max_hold_days"]:
                reason = "TIME"
            elif cfg["trend_break_exit"] and ret >= 0.05 and price < float(row["ma50"]):
                reason = "TRAIL_STOP"

            if reason:
                pnl_d  = pos["shares"] * (price - pos["entry_price"])
                equity += pos["cost"] + pnl_d
                trades.append({
                    "ticker":      ticker,
                    "entry_date":  pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date":   date.strftime("%Y-%m-%d"),
                    "days_held":   held,
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price":  round(price, 2),
                    "pnl_pct":     round(ret * 100, 2),
                    "pnl_dollar":  round(pnl_d, 2),
                    "exit_reason": reason,
                    "win":         ret > 0,
                    "year":        date.year,
                })
                del positions[ticker]

        # ── ENTRIES ───────────────────────────────────────────────
        if len(positions) < cfg["max_positions"]:
            for ticker, idf in inds.items():
                if ticker == "SPY" or ticker in positions:
                    continue
                if len(positions) >= cfg["max_positions"]:
                    break
                if date not in idf.index:
                    continue
                try:
                    row = idf.loc[date]
                    if is_entry(row, spy_r, spy_above_200, cfg):
                        pos_val = equity * cfg["position_size_pct"]
                        shares  = pos_val / float(row["Close"])
                        cost    = shares * float(row["Close"])
                        if cost > equity * 0.98:
                            continue
                        equity -= cost
                        positions[ticker] = {
                            "entry_price": float(row["Close"]),
                            "entry_date":  date,
                            "shares":      shares,
                            "cost":        cost,
                        }
                except Exception:
                    continue

        # ── MARK-TO-MARKET ────────────────────────────────────────
        open_val = 0.0
        for t, pos in positions.items():
            idf = inds.get(t)
            if idf is not None and date in idf.index:
                open_val += float(idf.loc[date, "Close"]) * pos["shares"]
        eq_curve.append({"date": date, "equity": equity + open_val})

    # Force-close remaining
    last = all_dates[-1]
    for ticker, pos in positions.items():
        idf   = inds.get(ticker)
        price = float(idf.loc[last, "Close"]) if (idf is not None and last in idf.index) else pos["entry_price"]
        ret   = (price - pos["entry_price"]) / pos["entry_price"]
        pnl_d = pos["shares"] * (price - pos["entry_price"])
        equity += pos["cost"] + pnl_d
        trades.append({
            "ticker":      ticker,
            "entry_date":  pos["entry_date"].strftime("%Y-%m-%d"),
            "exit_date":   last.strftime("%Y-%m-%d"),
            "days_held":   (last - pos["entry_date"]).days,
            "entry_price": round(pos["entry_price"], 2),
            "exit_price":  round(price, 2),
            "pnl_pct":     round(ret * 100, 2),
            "pnl_dollar":  round(pnl_d, 2),
            "exit_reason": "END",
            "win":         ret > 0,
            "year":        last.year,
        })

    return pd.DataFrame(trades), pd.DataFrame(eq_curve).set_index("date"), spy_df


# ══════════════════════════════════════════════════════════════════
#  STEP 5 — STATISTICS
# ══════════════════════════════════════════════════════════════════

def compute_stats(trades_df, equity_df, spy_df, cfg):
    wins   = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]
    n      = len(trades_df)

    eq       = equity_df["equity"]
    final_eq = eq.iloc[-1]
    spy_s    = spy_df[spy_df.index >= pd.Timestamp(cfg["start_date"])]["Close"]
    spy_roi  = (spy_s.iloc[-1] - spy_s.iloc[0]) / spy_s.iloc[0] * 100

    years    = 15
    cagr     = ((final_eq / cfg["starting_capital"]) ** (1 / years) - 1) * 100
    spy_cagr = ((spy_s.iloc[-1] / spy_s.iloc[0]) ** (1 / years) - 1) * 100

    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5) if daily_ret.std() > 0 else 0

    roll_max = eq.cummax()
    max_dd   = ((eq - roll_max) / roll_max * 100).min()

    gw = wins["pnl_dollar"].sum()
    gl = losses["pnl_dollar"].abs().sum()

    return {
        "total_trades":  n,
        "winners":       len(wins),
        "losers":        len(losses),
        "win_rate":      len(wins) / n * 100,
        "total_roi":     (final_eq - cfg["starting_capital"]) / cfg["starting_capital"] * 100,
        "cagr":          cagr,
        "spy_roi":       spy_roi,
        "spy_cagr":      spy_cagr,
        "avg_days":      trades_df["days_held"].mean(),
        "avg_win":       wins["pnl_pct"].mean() if len(wins) else 0,
        "avg_loss":      losses["pnl_pct"].mean() if len(losses) else 0,
        "profit_factor": gw / gl if gl > 0 else float("inf"),
        "max_drawdown":  max_dd,
        "sharpe":        sharpe,
        "final_equity":  final_eq,
        "equity_curve":  eq,
    }


# ══════════════════════════════════════════════════════════════════
#  STEP 6 — PRINT REPORT
# ══════════════════════════════════════════════════════════════════

def print_report(stats, trades_df, equity_df, cfg):
    s   = "═" * 62
    sep = "─" * 62
    print(f"\n{s}")
    print(f"  RESULTS  |  MINERVINI SEPA v3  |  2010–2024  (15 YEARS)")
    print(s)

    rows = [
        ("Total Trades",         f"{stats['total_trades']:,}"),
        ("Winners / Losers",     f"{stats['winners']:,} / {stats['losers']:,}"),
        ("Win Rate",             f"{stats['win_rate']:.1f}%"),
        ("",                     ""),
        ("Total ROI",            f"+{stats['total_roi']:.1f}%"),
        ("CAGR (Strategy)",      f"+{stats['cagr']:.1f}% / yr"),
        ("CAGR (SPY B&H)",       f"+{stats['spy_cagr']:.1f}% / yr"),
        ("Final Equity",         f"${stats['final_equity']:,.0f}"),
        ("",                     ""),
        ("Avg Days Held",        f"{stats['avg_days']:.1f} days"),
        ("Avg Winner",           f"+{stats['avg_win']:.1f}%"),
        ("Avg Loser",            f"{stats['avg_loss']:.1f}%"),
        ("Profit Factor",        f"{stats['profit_factor']:.2f}x"),
        ("Max Drawdown",         f"{stats['max_drawdown']:.1f}%"),
        ("Sharpe Ratio (ann.)",  f"{stats['sharpe']:.2f}"),
    ]
    for label, value in rows:
        if not label:
            print()
        else:
            print(f"  {label:<28} {value:>16}")

    # Exit breakdown
    print(f"\n{sep}")
    print("  EXIT REASONS")
    print(sep)
    for reason, cnt in trades_df["exit_reason"].value_counts().items():
        print(f"  {reason:<24} {cnt:>5,}  ({cnt/len(trades_df)*100:.1f}%)")

    # ── Year-by-year with ROI% ────────────────────────────────────
    # Build start-of-year equity from the equity curve
    eq = equity_df["equity"]
    yearly_eq_start = {}
    for yr in range(2010, 2025):
        yr_data = eq[eq.index.year == yr]
        if len(yr_data):
            yearly_eq_start[yr] = yr_data.iloc[0]

    print(f"\n{sep}")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>7} {'ROI%':>8} {'AvgP&L':>9} {'Net $':>12}  Market")
    print(sep)

    for yr, g in trades_df.groupby("year"):
        wr      = g["win"].mean() * 100
        avg_pnl = g["pnl_pct"].mean()
        net_d   = g["pnl_dollar"].sum()

        # Year ROI = net $ / equity at start of year
        eq_start = yearly_eq_start.get(yr, cfg["starting_capital"])
        roi_pct  = net_d / eq_start * 100

        # Was the market (SPY) positive this year?
        spy_yr   = "📈" if net_d >= 0 else "📉"

        print(f"  {yr:<6} {len(g):>7,} {wr:>6.1f}% {roi_pct:>+7.1f}% "
              f"{avg_pnl:>+8.1f}% {net_d:>+12,.0f}  {spy_yr}")

    # Top 10
    print(f"\n{sep}")
    print("  TOP 10 WINNING TRADES")
    print(sep)
    top = trades_df.nlargest(10, "pnl_pct")[
        ["ticker", "entry_date", "exit_date", "days_held", "pnl_pct", "exit_reason"]
    ]
    print(top.to_string(index=False))

    # v3 changes summary
    print(f"\n{sep}")
    print("  v3 CHANGES vs v2")
    print(sep)
    changes = [
        ("Market Filter",      "OFF  →  ON   (no trades when SPY < 200MA)"),
        ("Entry Confirmation", "1 day  →  2 consecutive closes above pivot"),
        ("Breakout Volume",    "2.0x  →  1.75x  (balance quality/quantity)"),
        ("Stop Loss",          "7%   →  8%   (less noise-driven stops)"),
        ("Yearly ROI%",        "Not shown  →  Now printed in report"),
    ]
    for label, value in changes:
        print(f"  {label:<24}  {value}")
    print(f"\n{s}\n")


# ══════════════════════════════════════════════════════════════════
#  STEP 7 — CHART
# ══════════════════════════════════════════════════════════════════

def plot_results(trades_df, equity_df, spy_df, stats, cfg):
    plt.style.use("dark_background")
    BG, SURF  = "#0a0c0f", "#111318"
    ACC, RED  = "#00e5a0", "#ff4d6d"
    GOLD, MUT = "#ffd166", "#5a6070"

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36)

    # ── Equity curve ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(SURF)
    eq  = equity_df["equity"]
    spy_s = spy_df[spy_df.index >= pd.Timestamp(cfg["start_date"])]["Close"]
    spy_s = spy_s / spy_s.iloc[0] * cfg["starting_capital"]

    # Shade bear market periods (SPY below 200MA)
    spy_200 = spy_df[spy_df.index >= pd.Timestamp(cfg["start_date"])]["ma200"]
    spy_close = spy_df[spy_df.index >= pd.Timestamp(cfg["start_date"])]["Close"]
    bear = spy_close < spy_200
    bear_starts = bear.index[bear & ~bear.shift(1).fillna(False)]
    bear_ends   = bear.index[~bear & bear.shift(1).fillna(False)]
    for bs in bear_starts:
        be = bear_ends[bear_ends > bs][0] if len(bear_ends[bear_ends > bs]) else eq.index[-1]
        ax1.axvspan(bs, be, alpha=0.08, color=RED, label="_nolegend_")

    ax1.fill_between(eq.index, cfg["starting_capital"], eq,
                     where=(eq >= cfg["starting_capital"]), alpha=0.15, color=ACC)
    ax1.fill_between(eq.index, cfg["starting_capital"], eq,
                     where=(eq < cfg["starting_capital"]), alpha=0.15, color=RED)
    ax1.plot(eq.index, eq, color=ACC, lw=1.8,
             label=f"SEPA v3  CAGR +{stats['cagr']:.1f}%/yr  (total +{stats['total_roi']:.0f}%)")
    ax1.plot(spy_s.index, spy_s, color=MUT, lw=1.2, ls="--",
             label=f"Buy & Hold SPY  CAGR +{stats['spy_cagr']:.1f}%/yr  (total +{stats['spy_roi']:.0f}%)")
    ax1.axhline(cfg["starting_capital"], color=MUT, lw=0.5, ls=":")

    # Red zone label
    ax1.text(0.01, 0.05, "▓ Red zones = SPY below 200MA (no new entries)",
             transform=ax1.transAxes, color=RED, fontsize=7, alpha=0.7)

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title("EQUITY CURVE — MINERVINI SEPA v3 vs. BUY & HOLD SPY  (2010–2024)",
                  color="white", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.tick_params(colors=MUT, labelsize=8)
    for sp in ax1.spines.values(): sp.set_color("#1e2430")

    # ── Win-rate donut ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(SURF)
    wr = stats["win_rate"]
    ax2.pie([wr, 100 - wr], colors=[ACC, "#1e2430"], startangle=90,
            wedgeprops={"width": 0.45, "edgecolor": SURF, "linewidth": 2})
    ax2.text(0, 0, f"{wr:.1f}%", ha="center", va="center",
             color="white", fontsize=18, fontweight="bold")
    ax2.set_title("WIN RATE", color=MUT, fontsize=9, fontweight="bold")

    # ── Hold duration histogram ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(SURF)
    ax3.hist(trades_df["days_held"], bins=20, color=ACC, alpha=0.7, edgecolor=SURF)
    ax3.axvline(stats["avg_days"], color=GOLD, lw=1.5, ls="--",
                label=f"Avg {stats['avg_days']:.1f}d")
    ax3.set_title("HOLD DURATION (days)", color=MUT, fontsize=9, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.tick_params(colors=MUT, labelsize=8)
    for sp in ax3.spines.values(): sp.set_color("#1e2430")

    # ── P&L distribution ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(SURF)
    ax4.hist(trades_df[trades_df["win"]]["pnl_pct"],  bins=20, color=ACC,
             alpha=0.7, label="Winners", edgecolor=SURF)
    ax4.hist(trades_df[~trades_df["win"]]["pnl_pct"], bins=15, color=RED,
             alpha=0.7, label="Losers",  edgecolor=SURF)
    ax4.axvline(0, color=MUT, lw=0.8)
    ax4.set_title("P&L DISTRIBUTION (%)", color=MUT, fontsize=9, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.tick_params(colors=MUT, labelsize=8)
    for sp in ax4.spines.values(): sp.set_color("#1e2430")

    # ── Year-by-year ROI % bars ───────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor(SURF)

    eq_curve = equity_df["equity"]
    yearly_eq_start = {}
    for yr in range(2010, 2025):
        yr_data = eq_curve[eq_curve.index.year == yr]
        if len(yr_data):
            yearly_eq_start[yr] = yr_data.iloc[0]

    yr_grp  = trades_df.groupby("year")
    years   = sorted(yr_grp.groups.keys())
    yr_roi  = []
    for y in years:
        net   = yr_grp.get_group(y)["pnl_dollar"].sum()
        start = yearly_eq_start.get(y, cfg["starting_capital"])
        yr_roi.append(net / start * 100)

    colors = [ACC if r >= 0 else RED for r in yr_roi]
    bars   = ax5.bar(years, yr_roi, color=colors, alpha=0.8, edgecolor=SURF, width=0.6)
    ax5.axhline(0, color=MUT, lw=0.8)
    ax5.set_title("ANNUAL ROI % (on start-of-year equity)", color=MUT, fontsize=9, fontweight="bold")
    ax5.tick_params(colors=MUT, labelsize=8)
    for sp in ax5.spines.values(): sp.set_color("#1e2430")
    for bar, val in zip(bars, yr_roi):
        ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 2.5
        ax5.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{val:+.0f}%", ha="center", va="bottom", color="white", fontsize=7)

    # ── Summary stats box ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(SURF)
    ax6.axis("off")
    stat_rows = [
        ("TOTAL TRADES",   f"{stats['total_trades']:,}",          "white"),
        ("WIN RATE",       f"{stats['win_rate']:.1f}%",            ACC),
        ("TOTAL ROI",      f"+{stats['total_roi']:.1f}%",          ACC),
        ("CAGR STRATEGY",  f"+{stats['cagr']:.1f}% / yr",          ACC),
        ("CAGR SPY",       f"+{stats['spy_cagr']:.1f}% / yr",      MUT),
        ("AVG DAYS HELD",  f"{stats['avg_days']:.1f}",             GOLD),
        ("AVG WINNER",     f"+{stats['avg_win']:.1f}%",            ACC),
        ("AVG LOSER",      f"{stats['avg_loss']:.1f}%",            RED),
        ("PROFIT FACTOR",  f"{stats['profit_factor']:.2f}x",       GOLD),
        ("MAX DRAWDOWN",   f"{stats['max_drawdown']:.1f}%",        RED),
        ("SHARPE RATIO",   f"{stats['sharpe']:.2f}",               GOLD),
        ("FINAL EQUITY",   f"${stats['final_equity']:,.0f}",       ACC),
    ]
    ax6.set_title("SUMMARY STATISTICS", color=MUT, fontsize=9, fontweight="bold")
    for i, (label, value, color) in enumerate(stat_rows):
        y = 1 - i * 0.082
        ax6.text(0.02, y, label, transform=ax6.transAxes, color=MUT,   fontsize=8, va="top")
        ax6.text(0.98, y, value, transform=ax6.transAxes, color=color, fontsize=8,
                 va="top", ha="right", fontweight="bold")

    plt.suptitle(
        "MINERVINI SEPA v3 — 15-YEAR BACKTEST (2010–2024)  |  Market Filter + Breakout Confirmation",
        color="white", fontsize=12, fontweight="bold", y=0.98,
    )
    plt.savefig(cfg["chart_file"], dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"📊 Chart saved → {cfg['chart_file']}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG

    data = download_all(cfg)
    if not data:
        print("❌ No data downloaded. Check your internet connection.")
        sys.exit(1)

    trades_df, equity_df, spy_df = run_backtest(data, cfg)

    if len(trades_df) == 0:
        print("❌ No trades generated. Try relaxing entry filters.")
        sys.exit(1)

    stats = compute_stats(trades_df, equity_df, spy_df, cfg)
    print_report(stats, trades_df, equity_df, cfg)

    trades_df.to_csv(cfg["trades_file"], index=False)
    print(f"📁 Trades CSV → {cfg['trades_file']}")

    plot_results(trades_df, equity_df, spy_df, stats, cfg)
    print(f"\n✅ Done! Open  {cfg['chart_file']}  to see results.\n")


if __name__ == "__main__":
    main()
