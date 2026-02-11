# models/neuralprophet_model.py

import numpy as np
import pandas as pd

from neuralprophet import NeuralProphet

from .base import BasePriceModel


class NeuralProphetPriceModel(BasePriceModel):
	def __init__(
		self,
		forecast_horizon: int = 144,  # 3일(30분단위) = 3 * 24 * 2 = 144
		n_lags: int = 240,           # 과거 5일(30분단위) = 5 * 24 * 2 = 240
	):
		# 하이퍼파라미터
		self.forecast_horizon = forecast_horizon
		self.n_lags = n_lags

		# RF / LGBM / LSTM 과 공통 필드
		self.model = None
		self.df: pd.DataFrame | None = None
		self.features: list[str] | None = None

		self.split_idx: int | None = None
		self.y_test: pd.Series | None = None
		self.y_pred: pd.Series | None = None
		self.rmse: float | None = None
		self.r2: float | None = None

		# NeuralProphet 전용 상태
		self.df_np: pd.DataFrame | None = None   # ds, y, GPT_Score
		self.trained_until: pd.Timestamp | None = None

	def _build_np_df(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		df_ml 을 받아 NeuralProphet용 DataFrame(ds, y, GPT_Score)으로 변환.

		- df: 최소한 date, price, (선택적으로) gpt_score 컬럼을 가진다고 가정
		- gpt_score 사용 여부는 self.features 기준으로 결정
		"""
		df_local = df.copy().sort_values("date")

		df_np = pd.DataFrame({
			"ds": df_local["date"],
			"y": df_local["price"].astype(float),
		})

		use_gpt = (
			"gpt_score" in df_local.columns
			and self.features is not None
			and "gpt_score" in self.features
		)

		if use_gpt:
			df_np["GPT_Score"] = df_local["gpt_score"].astype(float)
		else:
			df_np["GPT_Score"] = 0.0

		# ds, y 기준 결측 제거
		df_np = df_np.dropna(subset=["ds", "y"])

		return df_np

	def train(self, df: pd.DataFrame, features: list[str]):
		"""
		RandomForestPriceModel / LightGBMPriceModel 과 인터페이스를 맞춘 train.

		- df: feature + price + date 를 포함한 전체 ML용 데이터프레임
		- features: 학습에 사용할 feature 컬럼 리스트
		            (NeuralProphet에선 gpt_score 사용 여부 판단에 사용)
		"""
		# 공통 필드 저장 (RF/LGBM/LSTM과 동일 패턴)
		self.df = df
		self.features = features

		# NeuralProphet용 df 생성
		self.df_np = self._build_np_df(df)
		df_np = self.df_np
		self.trained_until = pd.to_datetime(df_np["ds"].max())

		# 시계열 split: 앞 80% train, 뒤 20% test (다른 모델들과 통일)
		n = len(df_np)
		self.split_idx = int(n * 0.8)

		df_train = df_np.iloc[: self.split_idx].copy()
		df_full = df_np.copy()  # 전체 구간 예측용

		# 모델 정의
		m = NeuralProphet(
			n_forecasts=self.forecast_horizon,
			n_lags=self.n_lags,
			n_changepoints=10,
			trend_reg=1.0,
			weekly_seasonality=True,
			daily_seasonality=True,
			yearly_seasonality=False,
			learning_rate=0.01,
			growth="off",
		)

		# GPT_Score regressor 등록
		m.add_future_regressor("GPT_Score")

		# 학습 (train 구간만 사용)
		m.fit(
			df_train,
			freq="30min",
			num_workers=0,
			checkpointing=False,
		)

		self.model = m

		# -------------------------------------------------
		# 테스트 구간 평가 (다른 모델들과 동일하게 "뒤 20%" 사용)
		# -------------------------------------------------
		try:
			from sklearn.metrics import mean_squared_error, r2_score

			# 전체 구간에 대해 예측 (history + pseudo-forecast)
			fc_hist = m.predict(df_full)

			# y: 실제 값
			y_all = fc_hist["y"].values

			# yhat1: 1-step-ahead 예측으로 간주
			if "yhat1" in fc_hist.columns:
				yhat1_all = fc_hist["yhat1"].values
			else:
				# 혹시 yhat1이 없는 구조라면, 첫 번째 yhat 계열을 사용
				yhat_cols = [c for c in fc_hist.columns if c.startswith("yhat")]
				if not yhat_cols:
					raise RuntimeError("NeuralProphet 결과에 yhat 계열 컬럼이 없습니다.")
				yhat1_all = fc_hist[yhat_cols[0]].values

			# split_idx 이후를 테스트 구간으로 사용
			test_start = self.split_idx
			y_test = y_all[test_start:]
			y_pred = yhat1_all[test_start:]

			if len(y_test) > 0:
				rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
				r2 = float(r2_score(y_test, y_pred))

				self.y_test = pd.Series(y_test)
				self.y_pred = pd.Series(y_pred)
				self.rmse = rmse
				self.r2 = r2
			else:
				self.y_test = pd.Series([], dtype=float)
				self.y_pred = pd.Series([], dtype=float)
				self.rmse = None
				self.r2 = None

		except Exception as e:
			print(f"[WARN] NeuralProphet backtest metric 실패: {e}")
			self.y_test = pd.Series([], dtype=float)
			self.y_pred = pd.Series([], dtype=float)
			self.rmse = None
			self.r2 = None

	def predict_test(self):
		"""
		RandomForestPriceModel / LightGBMPriceModel / LSTMPriceModel 과
		동일한 리턴형 유지.
		"""
		return (
			self.y_test,
			self.y_pred,
			self.split_idx,
			self.rmse,
			self.r2,
		)

	def predict_future(self, steps: int):
		"""
		향후 steps(30분 단위) 시점에 대한 예측선을 DataFrame으로 반환.

		- NeuralProphet의 n_forecasts(=forecast_horizon) 내에서
		  '대각선 추출' 방식으로 1-step, 2-step, ... 예측을 구성.
		- GPT_Score는 기본적으로 0.0으로 채운 상태에서 예측.
		"""
		if self.model is None or self.df_np is None:
			raise RuntimeError("NeuralProphet 모델이 학습되지 않았습니다. 먼저 train()을 호출하세요.")

		m: NeuralProphet = self.model
		df_np = self.df_np

		# 미래 GPT 스코어 (공지사항 없음 가정 → 0.0)
		future_regressors = pd.DataFrame({
			"GPT_Score": [0.0] * steps
		})

		# 과거 + 미래 dataframe 생성
		future = m.make_future_dataframe(
			df_np,
			periods=steps,
			n_historic_predictions=False,
			regressors_df=future_regressors,
		)

		forecast = m.predict(future)

		# y가 NaN인 행만 미래 구간
		future_rows = forecast[forecast["y"].isnull()].copy()

		# steps가 n_forecasts보다 크지 않다고 가정 (현재는 1일(48) 예측, n_forecasts=144)
		max_step = min(steps, self.forecast_horizon, len(future_rows))

		valid_dates = []
		valid_prices = []

		# 대각선 추출: i번째 예측시점 → 해당 row의 yhat{i}
		for i in range(1, max_step + 1):
			row_idx = future_rows.index[i - 1]
			col_name = f"yhat{i}"
			if col_name not in future_rows.columns:
				# 방어적으로, 없는 경우는 루프 종료
				break
			valid_dates.append(future_rows.loc[row_idx, "ds"])
			valid_prices.append(future_rows.loc[row_idx, col_name])

		forecast_df = pd.DataFrame({
			"date": valid_dates,
			"price": valid_prices,
		})

		return forecast_df
