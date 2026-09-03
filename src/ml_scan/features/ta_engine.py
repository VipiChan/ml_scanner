"""pandas_ta feature generation. Copied from Final Project `source/helper.py`."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

LITE_SPECS: list[tuple[str, dict]] = [
    ("rsi", {"length": 14}),
    ("atr", {"length": 14}),
    ("ema", {"length": 9}),
    ("ema", {"length": 21}),
    ("sma", {"length": 20}),
    ("macd", {}),
    ("bbands", {"length": 5, "std": 2.0}),
    ("adx", {"length": 14}),
    ("cci", {"length": 20}),
    ("willr", {"length": 14}),
    ("roc", {"length": 10}),
    ("obv", {}),
    ("mfi", {"length": 14}),
]


def get_callable_indicators() -> list[tuple[str, callable]]:
    """Scan all pandas_ta categories and return unique callable indicators."""
    all_indicators = ta.Category.keys()
    indicator_functions = []
    for category in all_indicators:
        for ind in ta.Category[category]:
            func = getattr(ta, ind, None)
            if func and callable(func):
                indicator_functions.append((ind, func))
    return list(dict.fromkeys(indicator_functions))


def compile_technical_features(df: pd.DataFrame) -> list:
    cores = ["open", "high", "low", "close", "volume"]
    inputs = {c: df[c] for c in cores if c in df.columns}
    generated_outputs = []
    for name, func in get_callable_indicators():
        try:
            code = func.__code__
            args = code.co_varnames[: code.co_argcount]
            kwargs = {c: inputs[c] for c in cores if c in args and c in inputs}
            if kwargs:
                res = func(**kwargs)
                if res is None:
                    continue
                if isinstance(res, pd.Series):
                    res.name = f"{name.upper()}"
                    generated_outputs.append(res)
                elif isinstance(res, pd.DataFrame):
                    generated_outputs.append(res)
        except Exception:
            continue
    return generated_outputs


def merge_and_clean_features(df: pd.DataFrame, generated_features: list) -> pd.DataFrame:
    if not generated_features:
        return df
    df_indicators = pd.concat(generated_features, axis=1)
    df_indicators.dropna(how="all", axis=1, inplace=True)
    merged_df = pd.concat([df, df_indicators], axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    core_cols = [c for c in ("open", "high", "low", "close") if c in merged_df.columns]
    if core_cols:
        merged_df.dropna(subset=core_cols, inplace=True)
    return merged_df


def generate_all_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrator: generate and merge every compatible pandas_ta indicator."""
    features = compile_technical_features(df)
    return merge_and_clean_features(df, features)


def generate_lite_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Small, fast indicator set used for smoke / e2e runs."""
    frame = df.copy()
    generated: list[pd.Series | pd.DataFrame] = []
    for name, kwargs in LITE_SPECS:
        func = getattr(ta, name, None)
        if func is None or not callable(func):
            continue
        try:
            res = func(
                open=frame["open"] if "open" in frame.columns else None,
                high=frame["high"] if "high" in frame.columns else None,
                low=frame["low"] if "low" in frame.columns else None,
                close=frame["close"] if "close" in frame.columns else None,
                volume=frame["volume"] if "volume" in frame.columns else None,
                **kwargs,
            )
        except TypeError:
            try:
                res = func(close=frame["close"], **kwargs)
            except Exception:
                continue
        except Exception:
            continue
        if res is None:
            continue
        if isinstance(res, pd.Series) and res.name is None:
            res.name = name.upper()
        generated.append(res)
    return merge_and_clean_features(frame, generated)


def ensure_atr(df: pd.DataFrame, period: int = 14, col: str = "ATR") -> pd.DataFrame:
    frame = df.copy()
    if col in frame.columns and frame[col].notna().any():
        return frame
    try:
        atr = ta.atr(high=frame["high"], low=frame["low"], close=frame["close"], length=period)
        if atr is not None:
            frame[col] = atr
            return frame
    except Exception:
        pass
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame[col] = tr.rolling(period, min_periods=period).mean()
    return frame
