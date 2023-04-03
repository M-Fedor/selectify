import numpy as np

from tensorflow.keras.utils import Sequence, to_categorical
from math import ceil, floor


class DataLoader(Sequence):

    def __init__(self,
        data: np.ndarray,
        batch_size: int,
        item_length: int,
        use_items_factor: float=1,
        shuffle: bool=True
    ) -> None:
        self.data = data
        self.data_size = floor(data.shape[0] * use_items_factor)
        self.batch_size = batch_size
        self.item_length = item_length
        self.shuffle = shuffle

        self.batch_holder = np.empty((self.batch_size, self.data.shape[1]), dtype=np.float32)
        self.item_indices = np.arange(0, self.data.shape[0])
        # self.on_epoch_end()

        self.item_indices = self.item_indices[0 : self.data_size]


    def __len__(self) -> int:
        return ceil(self.data_size / self.batch_size)


    def __getitem__(self, idx: int) -> np.ndarray:
        data_idx_start = idx * self.batch_size

        if data_idx_start + self.batch_size < self.data_size:
            data_idx_end = data_idx_start + self.batch_size
        else:
            data_idx_end = self.data_size

        
        if self.batch_holder.shape[0] != data_idx_end - data_idx_start:
            self.batch_holder = np.empty(
                (data_idx_end - data_idx_start, self.data.shape[1]),
                dtype=float
            )

        batch_indices = self.item_indices[data_idx_start : data_idx_end]
        self.batch_holder[:] = self.data[batch_indices]
        examples = self.batch_holder[:, 0 : self.item_length]
        labels = self.batch_holder[:, self.item_length]

        return examples, to_categorical(labels)


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.item_indices)
