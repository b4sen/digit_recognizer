import numpy as np 
import typing

class Batch:

	def __init__(self):

		self.data = None
		self.label = None 


class BatchGenerator:

	def __init__(self, data, batch_size: int, shuffle: bool):
		self.data = []
		self.labels = []
		for img, label in data:
			self.data.append(self.normalize(img,127.5))
			self.labels.append(label)
		self.data = np.array(self.data)
		self.labels = np.array(self.labels)
		self.batch_size = batch_size
		self.num_batches = int(np.ceil(self.data.shape[0] / self.batch_size))

	def __iter__(self) -> typing.Iterable[Batch]:
		for i in range(self.num_batches):
			start_index = i * self.batch_size
			stop_index = (i+1) * self.batch_size
			batch = Batch()
			batch.data = np.array(self.data[start_index:stop_index], dtype=np.float32)
			batch.label = self.labels[start_index:stop_index]
			yield batch

	def normalize(self, img, val):
		return (img-val)*(1/val)