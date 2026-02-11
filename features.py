# features.py
# 30분 기준 Feature Engineering

import pandas as pd


def filter_item(df_final: pd.DataFrame, target_keyword: str, target_grade: str | None):
	mask = df_final["name"].str.contains(target_keyword)

	if target_grade and target_grade != "전체":
		mask = mask & (df_final["grade"] == target_grade)

	df_target = df_final[mask].copy().sort_values("date")

	if len(df_target) == 0:
		return None

	top_item = df_target["name"].value_counts().idxmax()
	df_target = df_target[df_target["name"] == top_item]

	return df_target, top_item


# --------------------------------------------------
# Technical Indicators
# --------------------------------------------------
def calculate_rsi(series: pd.Series, window: int):
	delta = series.diff()
	gain = delta.clip(lower=0).rolling(window=window).mean()
	loss = -delta.clip(upper=0).rolling(window=window).mean()
	rs = gain / loss
	return 100 - (100 / (1 + rs))


def calculate_bollinger(series: pd.Series, window: int):
	sma = series.rolling(window=window).mean()
	std = series.rolling(window=window).std()
	return sma + (std * 2), sma - (std * 2)


# --------------------------------------------------
# Main Feature Generator (30분 기준)
# --------------------------------------------------
def make_ml_dataset(df_target: pd.DataFrame):
	df_ml = df_target.copy()

	# ==================================================
	# 1️⃣ Lag Features (30분 기준)
	# ==================================================
	df_ml["lag_30m"] = df_ml["price"].shift(1)
	df_ml["lag_1h"]  = df_ml["price"].shift(2)
	df_ml["lag_6h"]  = df_ml["price"].shift(12)
	df_ml["lag_24h"] = df_ml["price"].shift(48)

	# ==================================================
	# 2️⃣ RSI / Bollinger (시간 의미 보존)
	# ==================================================
	# 기존 10분 기준:
	# RSI 14 ≈ 2.3시간 → 30분 기준 ≈ 5
	# BB  20 ≈ 3.3시간 → 30분 기준 ≈ 7

	df_ml["rsi"] = calculate_rsi(df_ml["price"], window=5)
	df_ml["bb_upper"], df_ml["bb_lower"] = calculate_bollinger(
		df_ml["price"],
		window=7
	)

	# ==================================================
	# 3️⃣ 상태 파생 변수
	# ==================================================
	df_ml["is_overbought"] = (df_ml["price"] > df_ml["bb_upper"]).astype(int)
	df_ml["is_oversold"]  = (df_ml["price"] < df_ml["bb_lower"]).astype(int)

	# ==================================================
	# 4️⃣ 시간 정보
	# ==================================================
	df_ml["hour"] = df_ml["date"].dt.hour
	df_ml["day_of_week"] = df_ml["date"].dt.dayofweek

	# ==================================================
	# 5️⃣ 결측 제거
	# ==================================================
	df_ml = df_ml.dropna()

	# ==================================================
	# 6️⃣ Feature List
	# ==================================================
	features = [
		"lag_30m", "lag_1h", "lag_6h", "lag_24h",
		"rsi", "is_overbought", "is_oversold",
		"hour", "day_of_week",
	]

	if "gpt_score" in df_ml.columns:
		features.append("gpt_score")

	return df_ml, features
