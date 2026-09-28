import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def chronological_split(df, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy()
    )


def create_sequences(data, lookback):
    X = []
    y = []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])

    return np.asarray(X), np.asarray(y)


def prepare_data(df, columns, lookback=30):
    train_df, val_df, test_df = chronological_split(df)

    train_values = train_df[columns].astype(float).values
    val_values = val_df[columns].astype(float).values
    test_values = test_df[columns].astype(float).values

    scaler = MinMaxScaler()

    scaler.fit(train_values)

    train_scaled = scaler.transform(train_values)
    val_scaled = scaler.transform(val_values)
    test_scaled = scaler.transform(test_values)

    X_train, y_train = create_sequences(
        train_scaled,
        lookback
    )

    val_input = np.concatenate(
        [train_scaled[-lookback:], val_scaled],
        axis=0
    )

    X_val, y_val = create_sequences(
        val_input,
        lookback
    )

    test_input = np.concatenate(
        [val_scaled[-lookback:], test_scaled],
        axis=0
    )

    X_test, y_test = create_sequences(
        test_input,
        lookback
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        train_df,
        val_df,
        test_df
    )


def prepare_return_data(df, columns, lookback=30):
    data = df.copy()

    data["target_return"] = np.log(
        data["close"].shift(-1) / data["close"]
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna().reset_index(drop=True)

    train_df, val_df, test_df = chronological_split(data)

    train_features = train_df[columns].astype(float).values
    val_features = val_df[columns].astype(float).values
    test_features = test_df[columns].astype(float).values

    train_target = train_df["target_return"].astype(float).values
    val_target = val_df["target_return"].astype(float).values
    test_target = test_df["target_return"].astype(float).values

    scaler = MinMaxScaler()
    scaler.fit(train_features)

    train_scaled = scaler.transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)

    X_train = []
    y_train = []

    for i in range(lookback, len(train_scaled)):
        X_train.append(train_scaled[i - lookback:i])
        y_train.append(train_target[i])

    val_features_combined = np.concatenate(
        [train_scaled[-lookback:], val_scaled],
        axis=0
    )

    val_targets_combined = np.concatenate(
        [train_target[-lookback:], val_target],
        axis=0
    )

    X_val = []
    y_val = []

    for i in range(lookback, len(val_features_combined)):
        X_val.append(
            val_features_combined[i - lookback:i]
        )
        y_val.append(
            val_targets_combined[i]
        )

    test_features_combined = np.concatenate(
        [val_scaled[-lookback:], test_scaled],
        axis=0
    )

    test_targets_combined = np.concatenate(
        [val_target[-lookback:], test_target],
        axis=0
    )

    X_test = []
    y_test = []

    for i in range(lookback, len(test_features_combined)):
        X_test.append(
            test_features_combined[i - lookback:i]
        )
        y_test.append(
            test_targets_combined[i]
        )

    return (
        np.asarray(X_train),
        np.asarray(y_train),
        np.asarray(X_val),
        np.asarray(y_val),
        np.asarray(X_test),
        np.asarray(y_test),
        scaler,
        train_df,
        val_df,
        test_df
    )