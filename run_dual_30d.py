import argparse
import json

import pandas as pd

from core import backtest_engine, config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--symbols", type=str, default=",".join(config.SYMBOLS))
    p.add_argument("--direction", type=str, default=config.DEFAULT_DIRECTION)
    p.add_argument("--initial-equity", type=float, default=config.DEFAULT_INITIAL_EQUITY)
    p.add_argument("--margin", type=float, default=config.DEFAULT_MARGIN_PER_TRADE)
    p.add_argument("--leverage", type=float, default=config.DEFAULT_LEVERAGE)
    p.add_argument("--max-open", type=int, default=config.DEFAULT_MAX_OPEN_POSITIONS)
    p.add_argument("--tie-break", type=str, default="sl_first", choices=["sl_first", "tp_first"])
    p.add_argument("--dedup-min", type=int, default=config.DEDUP_WINDOW_MIN)
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print(f"Symbols: {symbols}")
    print(f"Days: {args.days} | Direction: {args.direction} | Dedup: {args.dedup_min} min")
    print("Strategies (config.STRATEGIES):")
    for s in config.STRATEGIES:
        print(f"  - {s['label']}: base={s['base_tf']} confirm={s['confirm_tf']} "
              f"ha={s['use_heikin_ashi']} ema={s['ema_fast']}/{s['ema_slow']} "
              f"tp={s['tp_pct']}% sl={s['sl_pct']}% dir={s['direction']}")
    print("-" * 70)

    result = backtest_engine.run_backtest_dual(
        symbols=symbols,
        strategies=config.STRATEGIES,
        direction=args.direction,
        initial_equity=args.initial_equity,
        margin_per_trade=args.margin,
        leverage=args.leverage,
        max_open_positions=args.max_open,
        days=args.days,
        tie_break=args.tie_break,
        dedup_window_min=args.dedup_min,
    )

    summary = result.get("summary", {})
    trades = result.get("trades", [])

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nTotal trades: {len(trades)}")

    if trades:
        df = pd.DataFrame(trades)
        print("\nBy strategy:")
        agg = {"trades": ("strategy", "count")}
        if "pnl_pct" in df.columns:
            agg["wins"] = ("pnl_pct", lambda x: (x > 0).sum())
        print(df.groupby("strategy").agg(**agg))
        if "exit_reason" in df.columns:
            print("\nBy exit_reason:")
            print(df["exit_reason"].value_counts())
        out_csv = f"run_dual_{args.days}d_trades.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved trades to: {out_csv}")

    out_summary = f"run_dual_{args.days}d_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Summary saved to: {out_summary}")


if __name__ == "__main__":
    main()
