"""Frozen copy of Final Project helper TA functions for parity tests."""

import pandas as pd
import pandas_ta as ta


def get_callable_indicators():
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
                if res is not None:
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
    return merge_and_clean_features(df, compile_technical_features(df))
