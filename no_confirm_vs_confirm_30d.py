"""
Сравнение на 30 днях, BTCUSDT:

  A) БАЗОВАЯ (с подтверждением, как в основной связке):
     - 4m/8m EMA(7/133) cross            TP0.7 / SL1.0
     - 2m/7m Heikin Ashi EMA(7/133) cross TP0.8 / SL0.9

  B) БЕЗ ПОДТВЕРЖДЕНИЯ (только кросс на базовом ТФ, без старшего ТФ):
     - 4m EMA(7/133) cross (без подтверждения на 8m)          TP0.7 / SL1.0
     - 2m Heikin Ashi EMA(7/133) cross (без подтверждения на 7m) TP0.8 / SL0.9

Всё остальное (EMA, TP/SL, dedup, margin, leverage, max_open) идентично,
чтобы разница результатов была именно из-за отсутствия подтверждения.

Артефакты: no_confirm_vs_confirm_30d_summary.txt и два CSV со сделками.
"""
import uuid
import pandas as pd

from core import backtest_engine

SYMBOLS = ["BTCUSDT"]
DAYS = 30
WARMUP_DAYS = 6
EMA_FAST = 7
EMA_SLOW = 133
INITIAL_EQUITY = 10000.0
MARGIN_PER_TRADE = 500.0
LEVERAGE = 50.0
MAX_OPEN_POSITIONS = 10
TIE_BREAK_SL_FIRST = True
DEDUP_WINDOW_MIN = 20

VARIANT_A_CONFIRMED = [
    {"name": "4m8m", "base": "4min", "confirm": "8min", "ha": False, "tp": 0.7, "sl": 1.0},
    {"name": "2m7m_ha", "base": "2min", "confirm": "7min", "ha": True, "tp": 0.8, "sl": 0.9},
]
VARIANT_B_NO_CONFIRM = [
    {"name": "4m_only", "base": "4min", "confirm": None, "ha": False, "tp": 0.7, "sl": 1.0},
    {"name": "2m_ha_only", "base": "2min", "confirm": None, "ha": True, "tp": 0.8, "sl": 0.9},
]


def resample_ohlc_right(df, rule):
    return df.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()


def to_heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha["close"].iloc[i - 1]) / 2.0)
    ha["open"] = ha_open
    ha["high"] = pd.concat([df["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha[["open", "high", "low", "close"]]


def map_htf_no_lookahead(base_index, htf_series):
    htf_df = htf_series.rename("val").reset_index()
    htf_df.columns = ["htf_time", "val"]
    base = pd.DataFrame({"base_time": base_index})
    merged = pd.merge_asof(
        base.sort_values("base_time"), htf_df.sort_values("htf_time"),
        left_on="base_time", right_on="htf_time", direction="backward",
    )
    return merged.set_index("base_time")["val"].reindex(base_index)


def compute_signals(df_base, df_confirm, fast, slow):
    out = df_base.copy()
    out["ema_fast"] = out["close"].ewm(span=fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    diff = out["ema_fast"] - out["ema_slow"]
    prev = diff.shift(1)
    out["cross_up"] = (prev <= 0) & (diff > 0)
    out["cross_down"] = (prev >= 0) & (diff < 0)

    if df_confirm is None:
        out["long_signal"] = out["cross_up"]
        out["short_signal"] = out["cross_down"]
        return out

    ema_fc = df_confirm["close"].ewm(span=fast, adjust=False).mean()
    ema_sc = df_confirm["close"].ewm(span=slow, adjust=False).mean()
    diffc = ema_fc - ema_sc
    prevc = diffc.shift(1)
    cu_c = (prevc <= 0) & (diffc > 0)
    cd_c = (prevc >= 0) & (diffc < 0)

    out["cross_up_confirm"] = map_htf_no_lookahead(out.index, cu_c.astype(float)).fillna(0).astype(bool)
    out["cross_down_confirm"] = map_htf_no_lookahead(out.index, cd_c.astype(float)).fillna(0).astype(bool)

    out["long_signal"] = out["cross_up"] & out["cross_up_confirm"]
    out["short_signal"] = out["cross_down"] & out["cross_down_confirm"]
    return out


def build_signals_for_strategy(raw_1m, strat, report_start_days):
    base_data, confirm_data = {}, {}
    report_start = None
    for sym, df1m in raw_1m.items():
        base = resample_ohlc_right(df1m, strat["base"])
        if strat["ha"]:
            base = to_heikin_ashi(base)
        confirm = None
        if strat["confirm"] is not None:
            confirm = resample_ohlc_right(df1m, strat["confirm"])
            if strat["ha"]:
                confirm = to_heikin_ashi(confirm)
        if len(base) < EMA_SLOW + 5:
            continue
        cutoff = base.index.max() - pd.Timedelta(days=report_start_days)
        report_start = cutoff if report_start is None else max(report_start, cutoff)
        base_data[sym] = base
        confirm_data[sym] = confirm

    signal_rows = []
    market_data = {}
    for sym in base_data:
        sigs = compute_signals(base_data[sym], confirm_data[sym], EMA_FAST, EMA_SLOW)
        market_data[sym] = sigs[["high", "low"]]
        cutoff_mask = sigs.index >= report_start
        for ts in sigs.index[sigs["long_signal"].fillna(False) & cutoff_mask]:
            signal_rows.append({"time": ts, "symbol": sym, "direction": "long",
                                 "entry_price": float(sigs.at[ts, "close"]), "strategy": strat["name"]})
        for ts in sigs.index[sigs["short_signal"].fillna(False) & cutoff_mask]:
            signal_rows.append({"time": ts, "symbol": sym, "direction": "short",
                                 "entry_price": float(sigs.at[ts, "close"]), "strategy": strat["name"]})

    return signal_rows, market_data, report_start


def dedup_signals(all_signals, window_min):
    df = pd.DataFrame(all_signals).sort_values("time").reset_index(drop=True)
    if df.empty:
        return df
    window = pd.Timedelta(minutes=window_min)
    kept_rows = []
    last_seen = {}
    for _, row in df.iterrows():
        key = (row["symbol"], row["direction"])
        prev_ts = last_seen.get(key)
        if prev_ts is not None and (row["time"] - prev_ts) <= window:
            continue
        kept_rows.append(row)
        last_seen[key] = row["time"]
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def simulate(market_data_by_strategy, signal_table, report_start_ts,
             margin, leverage, max_open, initial_equity, tp_sl_by_strat):
    signal_grouped = {ts: g for ts, g in signal_table.groupby("time")} if not signal_table.empty else {}

    all_index = set()
    for md in market_data_by_strategy.values():
        for df in md.values():
            all_index.update(df.index[df.index >= report_start_ts])
    all_times = pd.Index(sorted(all_index))

    open_positions = []
    equity = initial_equity
    trade_rows = []

    for ts in all_times:
        still_open = []
        for pos in open_positions:
            md = market_data_by_strategy[pos["strategy"]]
            df = md.get(pos["symbol"])
            if df is None or ts not in df.index:
                still_open.append(pos)
                continue
            row = df.loc[ts]
            entry_price = pos["entry_price"]
            tp_pct, sl_pct = tp_sl_by_strat[pos["strategy"]]
            if pos["direction"] == "long":
                stop_price = entry_price * (1.0 - sl_pct / 100.0)
                take_price = entry_price * (1.0 + tp_pct / 100.0)
                hit_sl = row["low"] <= stop_price
                hit_tp = row["high"] >= take_price
                exit_is_sl = TIE_BREAK_SL_FIRST if (hit_sl and hit_tp) else hit_sl
                exit_price = stop_price if exit_is_sl else take_price
                pnl_pct = (exit_price / entry_price) - 1.0
            else:
                stop_price = entry_price * (1.0 + sl_pct / 100.0)
                take_price = entry_price * (1.0 - tp_pct / 100.0)
                hit_sl = row["high"] >= stop_price
                hit_tp = row["low"] <= take_price
                exit_is_sl = TIE_BREAK_SL_FIRST if (hit_sl and hit_tp) else hit_sl
                exit_price = stop_price if exit_is_sl else take_price
                pnl_pct = 1.0 - (exit_price / entry_price)

            if hit_sl or hit_tp:
                notional = margin * leverage
                pnl_usdt = notional * pnl_pct
                equity += pnl_usdt
                trade_rows.append({
                    "strategy": pos["strategy"], "symbol": pos["symbol"], "direction": pos["direction"],
                    "entry_time": pos["entry_time"].isoformat(), "exit_time": ts.isoformat(),
                    "entry_price": entry_price, "exit_price": float(exit_price),
                    "exit_reason": "SL" if exit_is_sl else "TP",
                    "pnl_pct": pnl_pct * 100.0, "pnl_usdt": pnl_usdt,
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        if ts in signal_grouped:
            for _, sig in signal_grouped[ts].iterrows():
                if len(open_positions) >= max_open:
                    continue
                open_positions.append({
                    "id": str(uuid.uuid4()), "symbol": sig["symbol"], "direction": sig["direction"],
                    "strategy": sig["strategy"], "entry_time": ts, "entry_price": float(sig["entry_price"]),
                })

    return trade_rows, equity


def run_variant(raw_1m, strats, label):
    all_signals = []
    market_data_by_strategy = {}
    report_starts = []
    for strat in strats:
        sigs, market, rs = build_signals_for_strategy(raw_1m, strat, DAYS)
        all_signals.extend(sigs)
        market_data_by_strategy[strat["name"]] = market
        report_starts.append(rs)
        print(f"  [{label}] {strat['name']}: {len(sigs)} raw signals")
    report_start = min(report_starts)

    deduped = dedup_signals(all_signals, DEDUP_WINDOW_MIN)
    dropped = len(all_signals) - len(deduped)
    print(f"  [{label}] raw={len(all_signals)} -> after dedup={len(deduped)} (dropped {dropped})")

    tp_sl_by_strat = {s["name"]: (s["tp"], s["sl"]) for s in strats}
    trades, final_equity = simulate(
        market_data_by_strategy, deduped, report_start,
        MARGIN_PER_TRADE, LEVERAGE, MAX_OPEN_POSITIONS, INITIAL_EQUITY, tp_sl_by_strat,
    )

    wins = sum(1 for t in trades if t["pnl_usdt"] > 0)
    losses = sum(1 for t in trades if t["pnl_usdt"] < 0)
    net_pnl = final_equity - INITIAL_EQUITY
    winrate = (wins / len(trades) * 100.0) if trades else 0.0
    gross_win = sum(t["pnl_usdt"] for t in trades if t["pnl_usdt"] > 0)
    gross_loss = abs(sum(t["pnl_usdt"] for t in trades if t["pnl_usdt"] < 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    eq = INITIAL_EQUITY
    curve = [eq]
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        eq += t["pnl_usdt"]
        curve.append(eq)
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        "label": label, "trades": trades, "n_trades": len(trades), "wins": wins, "losses": losses,
        "winrate": winrate, "net_pnl": net_pnl, "pf": pf, "max_dd": max_dd,
        "raw_signals": len(all_signals), "kept_signals": len(deduped),
    }


def main():
    print(f"Fetching {DAYS}d + {WARMUP_DAYS}d warmup for {SYMBOLS}...", flush=True)
    raw_1m = {}
    for sym in SYMBOLS:
        df = backtest_engine.ensure_backtest_history(sym, days=DAYS, warmup_days=WARMUP_DAYS)
        if df is not None and not df.empty:
            raw_1m[sym] = df
    print(f"Got data for {len(raw_1m)}/{len(SYMBOLS)} symbols.\n", flush=True)

    res_a = run_variant(raw_1m, VARIANT_A_CONFIRMED, "A_confirmed")
    res_b = run_variant(raw_1m, VARIANT_B_NO_CONFIRM, "B_no_confirm")

    lines = []
    lines.append(f"=== A vs B, {DAYS}d, BTCUSDT ===\n")
    lines.append(f"{'metric':<18}{'A_confirmed':>15}{'B_no_confirm':>15}")
    for key, fmt in [("raw_signals", "{:d}"), ("kept_signals", "{:d}"), ("n_trades", "{:d}"),
                     ("wins", "{:d}"), ("losses", "{:d}"), ("winrate", "{:.2f}"),
                     ("net_pnl", "{:.2f}"), ("pf", "{:.3f}"), ("max_dd", "{:.2f}")]:
        va = fmt.format(res_a[key])
        vb = fmt.format(res_b[key])
        lines.append(f"{key:<18}{va:>15}{vb:>15}")

    summary = "\n".join(lines)
    print("\n" + summary)

    with open("no_confirm_vs_confirm_30d_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    if res_a["trades"]:
        pd.DataFrame(res_a["trades"]).sort_values("entry_time").to_csv(
            "no_confirm_vs_confirm_30d_A_confirmed_trades.csv", index=False)
    if res_b["trades"]:
        pd.DataFrame(res_b["trades"]).sort_values("entry_time").to_csv(
            "no_confirm_vs_confirm_30d_B_no_confirm_trades.csv", index=False)

    print("\nSaved: no_confirm_vs_confirm_30d_summary.txt, "
          "no_confirm_vs_confirm_30d_A_confirmed_trades.csv, "
          "no_confirm_vs_confirm_30d_B_no_confirm_trades.csv")


if __name__ == "__main__":
    main()
