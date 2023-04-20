import numpy as np
from math import floor

train_data_path = '/home/mfedor/covid_training_data/long_few_barcodes_chunks_3_low_quality/training_data.npy'
output_training_data_path = '/home/mfedor/covid_training_data/long_few_barcodes_chunks_3_low_quality/training_data_part.npy'
output_validation_data_path = '/home/mfedor/covid_training_data/long_few_barcodes_chunks_3_low_quality/validation_data_part.npy'
validation_data_factor = 0.2

train_data = np.load(train_data_path, allow_pickle=True, mmap_mode='r')
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)

validation_data_size = floor(train_data.shape[0] * validation_data_factor)
validation_data = train_data[0 : validation_data_size]
train_data = train_data[validation_data_size : train_data.shape[0]]

np.save(output_training_data_path, train_data)
np.save(output_validation_data_path, validation_data)
