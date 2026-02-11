# preprocess.py
import pandas as pd
import numpy as np
from datetime import timedelta

# =========================================================
# 1. 원본 가격 데이터 정제 + 30분봉 변환	# 사용 안함
# =========================================================
# def preprocess_ohlc_and_fill(df_raw: pd.DataFrame) -> pd.DataFrame:
# 	"""
# 	- 10분 단위 raw 가격 로그를 입력으로 받아
# 	- 원본 단계에서 1차 이상치 제거
# 	- 30분봉(Open/High/Low/Close/Mean)으로 변환
# 	- 결측 구간 ffill/bfill 처리
# 	"""

# 	df = df_raw.copy()

# 	# datetime index 설정
# 	if "logged_at" in df.columns:
# 		df["logged_at"] = pd.to_datetime(df["logged_at"])
# 		df = df.set_index("logged_at")

# 	# -------------------------------
# 	# 1차 이상치 제거 (raw 단계)
# 	# -------------------------------
# 	raw_window = 432   # 약 3일
# 	raw_sigma = 7

# 	rolling_mean = df["current_min_price"].rolling(
# 		window=raw_window, center=True
# 	).mean()
# 	rolling_std = df["current_min_price"].rolling(
# 		window=raw_window, center=True
# 	).std()

# 	upper = rolling_mean + raw_sigma * rolling_std
# 	lower = rolling_mean - raw_sigma * rolling_std

# 	outliers = (
# 		(df["current_min_price"] > upper) |
# 		(df["current_min_price"] < lower)
# 	)

# 	if outliers.any():
# 		df.loc[outliers, "current_min_price"] = np.nan
# 		df["current_min_price"] = df["current_min_price"].interpolate(method="linear")

# 	# -------------------------------
# 	# 30분봉 리샘플링
# 	# -------------------------------
# 	df_resampled = (
# 		df["current_min_price"]
# 		.resample("30min")
# 		.agg(["first", "max", "min", "last", "mean"])
# 	)

# 	df_resampled.columns = ["Open", "High", "Low", "Close", "Price_Mean"]

# 	# 결측 구간 보정
# 	df_resampled = df_resampled.ffill().bfill()

# 	return df_resampled


# =========================================================
# 2. Rolling Z-Score 기반 이상치 제거
# =========================================================
def clean_outliers_rolling(
	df: pd.DataFrame,
	column: str = "Price_Mean",
	window: int = 48,
	sigma: float = 3.0,
) -> pd.DataFrame:
	"""
	- rolling mean/std 기반 Z-score 방식 이상치 제거
	- 이상치는 NaN 처리 후 선형 보간
	"""

	df_clean = df.copy()

	rolling_mean = df_clean[column].rolling(window=window, center=True).mean()
	rolling_std = df_clean[column].rolling(window=window, center=True).std()

	upper = rolling_mean + sigma * rolling_std
	lower = rolling_mean - sigma * rolling_std

	outliers = (df_clean[column] > upper) | (df_clean[column] < lower)

	if outliers.any():
		df_clean.loc[outliers, column] = np.nan
		df_clean[column] = df_clean[column].interpolate(method="linear")

	return df_clean


# =========================================================
# 3. GPT 공지 점수 매핑
# =========================================================
def apply_gpt_scores(
	df_price: pd.DataFrame,
	df_gpt: pd.DataFrame | None,
	score_col: str = "GPT_Score",
) -> pd.DataFrame:
	"""
	- df_price: datetime index를 가진 가격 데이터
	- df_gpt: notice_date, gpt_score 컬럼을 가진 공지 데이터
	- 공지일 10:00 ~ +7일 06:00 구간에 gpt_score 적용
	"""

	df = df_price.copy()
	df[score_col] = 0.0

	if df_gpt is None or df_gpt.empty:
		return df

	for _, row in df_gpt.iterrows():
		notice_date = pd.to_datetime(row["notice_date"])
		score = row["gpt_score"]

		start = notice_date.replace(hour=10, minute=0, second=0)
		end = (notice_date + timedelta(days=7)).replace(hour=6, minute=0, second=0)

		mask = (df.index >= start) & (df.index < end)
		if mask.any():
			df.loc[mask, score_col] = score

	return df


def resample_to_30min_for_app(df: pd.DataFrame) -> pd.DataFrame:
	"""
	앱에서 사용하는 df_target (date, price, name, grade, item_id ...) 를
	30분 단위로 resample 해서 반환한다.

	- price: 30분 평균
	- 나머지 컬럼은 '마지막 값' 기준으로 채움 (이름/등급 등은 어차피 고정)
	"""

	df_res = df.copy()
	df_res["date"] = pd.to_datetime(df_res["date"])

	# 1) 숫자 컬럼(특히 price) → 30분 평균
	price_agg = (
		df_res
		.set_index("date")["price"]
		.resample("30min")
		.mean()
		.to_frame()
	)

	price_agg = price_agg.rename(columns={"price": "price"})

	# 2) 기타 컬럼은 마지막 값 기준으로 forward-fill 후 resample 마지막 값 사용
	other_cols = [c for c in df_res.columns if c not in ["date", "price"]]

	if other_cols:
		others = (
			df_res
			.set_index("date")[other_cols]
			.resample("30min")
			.last()
			.ffill()
		)
		df_30 = price_agg.join(others, how="left")
	else:
		df_30 = price_agg

	df_30 = df_30.reset_index()  # date 컬럼 복구
	return df_30
