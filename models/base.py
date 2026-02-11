from abc import ABC, abstractmethod


class BasePriceModel(ABC):

	@abstractmethod
	def train(self, df, features):
		pass

	@abstractmethod
	def predict_test(self):
		pass

	@abstractmethod
	def predict_future(self, steps: int):
		pass
