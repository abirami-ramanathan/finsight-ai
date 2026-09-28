import numpy as np
import pandas as pd
import ta
from arch import arch_model


def add_garch_volatility(df, return_column="return_1d"):
    returns = df[return_column].astype(float) * 100
    valid_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    volatility = pd.Series(np.nan, index=df.index, dtype=float)

    if len(valid_returns) >= 100:
        try:
            model = arch_model(
                valid_returns,
                mean="Zero",
                vol="GARCH",
                p=1,
                q=1,
                rescale=False
            )
            result = model.fit(disp="off")
            conditional_volatility = result.conditional_volatility / 100
            volatility.loc[conditional_volatility.index] = conditional_volatility
        except Exception:
            volatility = df[return_column].rolling(20).std()
    else:
        volatility = df[return_column].rolling(20).std()

    df["garch_style_vol"] = volatility
    return df


def add_causal_garch_volatility(df, train_ratio=0.70):
    df = df.copy()

    returns = df["return_1d"].astype(float).replace(
        [np.inf, -np.inf],
        np.nan
    )

    n = len(df)
    train_end = int(n * train_ratio)

    train_returns = returns.iloc[:train_end].dropna() * 100

    if len(train_returns) < 100:
        df["garch_style_vol"] = returns.rolling(20).std()
        return df

    model = arch_model(
        train_returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        rescale=False
    )

    result = model.fit(disp="off")

    params = result.params

    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])

    conditional = result.conditional_volatility

    volatility = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    volatility.loc[conditional.index] = (
        conditional / 100
    )

    last_variance = (
        float(conditional.iloc[-1]) ** 2
    )

    previous_return = float(
        train_returns.iloc[-1]
    )

    for i in range(train_end, n):
        variance = (
            omega
            + alpha * (previous_return ** 2)
            + beta * last_variance
        )

        variance = max(variance, 1e-12)

        volatility.iloc[i] = np.sqrt(variance) / 100

        current_return = returns.iloc[i]

        if pd.notna(current_return):
            previous_return = float(current_return * 100)

        last_variance = variance

    df["garch_style_vol"] = volatility

    return df


def build_features(df, asset_type="stock"):
    df = df.copy()

    df.columns = [str(col).lower() for col in df.columns]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df = df.sort_index()

    df["return_1d"] = df["close"].pct_change()

    df["ema_12"] = ta.trend.EMAIndicator(
        close=df["close"],
        window=12
    ).ema_indicator()

    df["rsi_14"] = ta.momentum.RSIIndicator(
        close=df["close"],
        window=14
    ).rsi()

    df["mfi_14"] = ta.volume.MFIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=14
    ).money_flow_index()

    stochastic = ta.momentum.StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        smooth_window=3
    )

    df["stoch_k"] = stochastic.stoch()
    df["stoch_d"] = stochastic.stoch_signal()

    bollinger = ta.volatility.BollingerBands(
        close=df["close"],
        window=20,
        window_dev=2
    )

    df["bb_high"] = bollinger.bollinger_hband()
    df["bb_low"] = bollinger.bollinger_lband()

    macd = ta.trend.MACD(
        close=df["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["volatility_20"] = (
        df["return_1d"].rolling(20).std()
    )

    df = add_garch_volatility(
        df,
        "return_1d"
    )

    if asset_type == "crypto":
        df["volatility_5"] = (
            df["return_1d"].rolling(5).std()
        )

    if asset_type == "reit":
        if "dividend_yield" not in df.columns:
            df["dividend_yield"] = np.nan

        if "occupancy_rate" not in df.columns:
            df["occupancy_rate"] = np.nan

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.ffill().bfill()
    df = df.dropna()

    return df


def add_features(df, asset_type="stock"):
    return build_features(df, asset_type)


def feature_columns(asset_type="stock"):
    columns = [
        "close",
        "volume",
        "return_1d",
        "ema_12",
        "rsi_14",
        "mfi_14",
        "stoch_k",
        "stoch_d",
        "bb_high",
        "bb_low",
        "macd",
        "macd_signal",
        "volatility_20",
        "garch_style_vol"
    ]

    if asset_type == "crypto":
        columns.append("volatility_5")

    if asset_type == "reit":
        columns.extend([
            "dividend_yield",
            "occupancy_rate"
        ])

    return columns