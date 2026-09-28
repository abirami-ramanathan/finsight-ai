from pathlib import Path
import pandas as pd


class ProjectContext:
    def __init__(self, root_dir=None):
        self.root = Path(root_dir or ".")

        self.forecasting_file = (
            self.root
            / "outputs"
            / "final_forecasting_results.csv"
        )

        self.recommendation_file = (
            self.root
            / "outputs"
            / "recommendation"
            / "recommendation_assets.csv"
        )

        self.bandit_file = (
            self.root
            / "outputs"
            / "recommendation"
            / "recommendation_evaluation.csv"
        )

        self.forecasting = self._load(
            self.forecasting_file
        )

        self.recommendations = self._load(
            self.recommendation_file
        )

        self.bandit = self._load(
            self.bandit_file
        )

    def _load(self, path):
        if not path.exists():
            return pd.DataFrame()

        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _symbol_series(self):
        if "symbol" in self.forecasting.columns:
            return (
                self.forecasting["symbol"]
                .fillna("")
                .astype(str)
                .str.upper()
            )

        return pd.Series(
            "",
            index=self.forecasting.index
        )

    def _find_asset(self, query):
        q = query.lower()

        patterns = {
            "BTC": [
                "btc",
                "bitcoin"
            ],
            "ETH": [
                "eth",
                "ethereum"
            ],
            "ITC_NS": [
                "itc"
            ],
            "TCS_NS": [
                "tcs"
            ],
            "INFY_NS": [
                "infy",
                "infosys"
            ],
            "RELIANCE_NS": [
                "reliance"
            ],
            "HDFCBANK_NS": [
                "hdfc bank",
                "hdfcbank"
            ],
            "HCLTECH_NS": [
                "hcl tech",
                "hcltech"
            ],
            "EMBASSY": [
                "embassy"
            ],
            "MINDSPACE": [
                "mindspace"
            ],
            "BIRET": [
                "biret"
            ]
        }

        for asset, keywords in patterns.items():
            for keyword in keywords:
                if keyword in q:
                    return asset

        return None

    def module_a_answer(self, query):
        q = query.lower()

        features = [
            (
                "EMA-12",
                "Exponential Moving Average captures the recent price trend while giving greater weight to recent observations."
            ),
            (
                "RSI-14",
                "Relative Strength Index captures momentum and the strength of recent price movements."
            ),
            (
                "MFI-14",
                "Money Flow Index combines price and volume information to represent buying and selling pressure."
            ),
            (
                "Stochastic K/D",
                "Stochastic indicators describe the current price relative to its recent trading range."
            ),
            (
                "Bollinger Bands",
                "Bollinger Bands describe price movement relative to a moving average and market volatility."
            ),
            (
                "MACD",
                "MACD captures relationships between moving averages and provides information about trend and momentum."
            ),
            (
                "Rolling Volatility",
                "Rolling volatility represents the changing variability of asset prices."
            ),
            (
                "GARCH-style Volatility",
                "GARCH-style volatility provides an additional representation of changing market volatility."
            )
        ]

        selected = features

        if "rsi" in q:
            selected = [
                item for item in features
                if "RSI" in item[0]
            ]

        elif "macd" in q:
            selected = [
                item for item in features
                if "MACD" in item[0]
            ]

        elif "ema" in q:
            selected = [
                item for item in features
                if "EMA" in item[0]
            ]

        elif "mfi" in q:
            selected = [
                item for item in features
                if "MFI" in item[0]
            ]

        elif "stochastic" in q:
            selected = [
                item for item in features
                if "Stochastic" in item[0]
            ]

        elif "bollinger" in q:
            selected = [
                item for item in features
                if "Bollinger" in item[0]
            ]

        lines = [
            "FinSight AI uses multiple technical and market features "
            "to represent trend, momentum, trading pressure and volatility.",
            ""
        ]

        for name, explanation in selected:
            lines.append(
                f"{name}: {explanation}"
            )

        lines.extend([
            "",
            "The purpose of combining these features is to provide the "
            "forecasting models with different views of market behaviour "
            "rather than relying only on the closing price.",
            "",
            "Important XAI distinction: these are the input features used "
            "by the forecasting pipeline. SHAP and Integrated Gradients "
            "are used separately in Module E to determine which features "
            "contributed most strongly to an individual model prediction."
        ])

        return "\n".join(lines)

    def module_b_answer(self, query):
        if self.forecasting.empty:
            return (
                "No forecasting results are available in "
                "final_forecasting_results.csv."
            )

        q = query.lower()

        df = self.forecasting.copy()

        symbols = self._symbol_series()

        asset = self._find_asset(query)

        if asset:
            matches = df[
                symbols.str.contains(
                    asset,
                    case=False,
                    na=False
                )
            ].copy()
        else:
            matches = df.copy()

        if matches.empty:
            return (
                "No forecasting result was found for the requested asset."
            )

        for column in [
            "r2",
            "mae",
            "rmse",
            "mape"
        ]:
            if column in matches.columns:
                matches[column] = pd.to_numeric(
                    matches[column],
                    errors="coerce"
                )

        model_question = any(
            phrase in q
            for phrase in [
                "performed well",
                "performed best",
                "best model",
                "which model",
                "better model",
                "best forecasting",
                "best prediction"
            ]
        )

        if model_question and "r2" in matches.columns:
            best = matches.loc[
                matches["r2"].idxmax()
            ]

            asset_name = str(
                best.get(
                    "symbol",
                    best.get(
                        "scheme_code",
                        "unknown"
                    )
                )
            )

            model_name = str(
                best.get(
                    "model",
                    "unknown"
                )
            )

            r2 = float(best["r2"])

            mae = (
                float(best["mae"])
                if pd.notna(best.get("mae"))
                else None
            )

            rmse = (
                float(best["rmse"])
                if pd.notna(best.get("rmse"))
                else None
            )

            mape = (
                float(best["mape"])
                if pd.notna(best.get("mape"))
                else None
            )

            lines = [
                f"For {asset_name}, the best evaluated model by R² is {model_name}.",
                "",
                "Numerical evidence:"
            ]

            lines.append(
                f"R² = {r2:.4f}"
            )

            if mae is not None:
                lines.append(
                    f"MAE = {mae:.2f}"
                )

            if rmse is not None:
                lines.append(
                    f"RMSE = {rmse:.2f}"
                )

            if mape is not None:
                lines.append(
                    f"MAPE = {mape:.2f}%"
                )

            lines.extend([
                "",
                "What this means:",
                f"The R² value of {r2:.4f} indicates that the model explains "
                "a large proportion of the variation in the evaluated target.",
            ])

            if mape is not None:
                lines.append(
                    f"The MAPE of {mape:.2f}% represents the average prediction "
                    "error in percentage terms; lower values indicate smaller "
                    "percentage errors."
                )

            if mae is not None:
                lines.append(
                    f"The MAE of {mae:.2f} represents the average absolute "
                    "prediction error in the evaluation units."
                )

            if rmse is not None:
                lines.append(
                    f"The RMSE of {rmse:.2f} gives greater weight to larger "
                    "prediction errors."
                )

            if model_name.lower() == "gru_vol":
                lines.extend([
                    "",
                    "Why this model is relevant:",
                    "GRU+Vol extends the GRU sequence model with volatility "
                    "information, allowing the forecasting pipeline to "
                    "represent changing market variability."
                ])

            lines.extend([
                "",
                "XAI connection:",
                "These metrics explain how well the model performed, but "
                "they do not by themselves explain which input features "
                "caused an individual prediction. That feature-level "
                "explanation belongs to Module E using SHAP or Integrated "
                "Gradients."
            ])

            return "\n".join(lines)

        lines = [
            "FinSight AI forecasting evidence:",
            ""
        ]

        for _, row in matches.iterrows():

            asset_name = str(
                row.get(
                    "symbol",
                    row.get(
                        "scheme_code",
                        "unknown"
                    )
                )
            )

            model_name = str(
                row.get(
                    "model",
                    "unknown"
                )
            )

            lines.append(
                f"{asset_name} — {model_name}"
            )

            if pd.notna(row.get("r2")):
                lines.append(
                    f"R²: {float(row['r2']):.4f}"
                )

            if pd.notna(row.get("mae")):
                lines.append(
                    f"MAE: {float(row['mae']):.2f}"
                )

            if pd.notna(row.get("rmse")):
                lines.append(
                    f"RMSE: {float(row['rmse']):.2f}"
                )

            if pd.notna(row.get("mape")):
                lines.append(
                    f"MAPE: {float(row['mape']):.2f}%"
                )

            lines.append("")

        lines.extend([
            "Interpretation:",
            "Higher R² indicates stronger explanatory performance, "
            "while lower MAE, RMSE and MAPE indicate smaller prediction errors.",
            "",
            "These results come directly from the trained forecasting "
            "experiments stored by the FinSight AI pipeline."
        ])

        return "\n".join(lines)

    def module_c_answer(self, query):
        q = query.lower()

        lines = [
            "FinSight AI uses a two-stage personalized recommendation pipeline.",
            "",
            "1. KNN candidate retrieval:",
            "KNN compares the investor context with asset profiles and "
            "retrieves the five most similar candidates.",
            "",
            "2. LinUCB contextual reranking:",
            "LinUCB evaluates those candidates using contextual bandit "
            "scores and balances exploitation with exploration.",
            "",
            "Exploitation means giving more attention to candidates that "
            "have produced stronger observed rewards.",
            "Exploration means testing alternatives where the model has "
            "greater uncertainty.",
            "",
            "3. Feedback:",
            "When a reward is observed, LinUCB updates its parameters for "
            "the selected candidate. This allows the recommendation policy "
            "to adapt over repeated interactions."
        ]

        asks_results = any(
            phrase in q
            for phrase in [
                "result",
                "results",
                "performance",
                "metric",
                "metrics",
                "evaluation",
                "2000",
                "reward",
                "regret"
            ]
        )

        if asks_results and not self.bandit.empty:

            overall = self.bandit[
                self.bandit["cluster"]
                .astype(str)
                .str.lower()
                == "overall"
            ]

            if not overall.empty:

                row = overall.iloc[0]

                rounds = int(
                    row["rounds"]
                )

                mean_reward = float(
                    row["mean_reward"]
                )

                mean_regret = float(
                    row["mean_regret"]
                )

                cumulative_reward = float(
                    row["cumulative_reward"]
                )

                cumulative_regret = float(
                    row["cumulative_regret"]
                )

                lines.extend([
                    "",
                    "OFFLINE EVALUATION EVIDENCE:",
                    f"Evaluation rounds: {rounds}",
                    f"Mean reward: {mean_reward:.4f}",
                    f"Mean regret: {mean_regret:.4f}",
                    f"Cumulative reward: {cumulative_reward:.2f}",
                    f"Cumulative regret: {cumulative_regret:.2f}",
                    "",
                    "Interpretation:",
                    f"Across {rounds} simulated offline rounds, the "
                    f"average reward was {mean_reward:.4f}.",
                    f"The mean regret was {mean_regret:.4f}, representing "
                    "the average reward gap relative to the best available "
                    "candidate in the simulated evaluation.",
                    "",
                    "Important limitation:",
                    "These are offline simulated bandit results, not real "
                    "user engagement measurements."
                ])

        return "\n".join(lines)

    def get_context(self, query, module):
        if module == "A":
            return self.module_a_answer(query)

        if module == "B":
            return self.module_b_answer(query)

        if module == "C":
            return self.module_c_answer(query)

        return ""