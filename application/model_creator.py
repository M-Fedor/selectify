import numpy as np
import os
import random
import sys
import tensorflow as tf

from keras import backend as K

from classification import SarsCoV2Classifier


SEED = 13

def reset_random_generators():
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


#session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
#K.set_session(session)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

classifier = SarsCoV2Classifier(
    train_data_path='/home/mfedor/covid_training_data/training_data_part.npy',
    validation_data_path='/home/mfedor/covid_training_data/validation_data_part.npy',
    signal_length=3_000,
    signal_begin=2_000,
    train_batch_size=2_048,
    validation_batch_size=32
)

reset_random_generators()

classifier.initialize_training(
    strides=7,
    kernel_size=60,
    use_data_factor=1,
    shuffle=True
)

classifier.summarize_keras_trainable_variables()
classifier.train(epochs=100)
recall, specificity, precision, f1_score, accuracy = classifier.evaluate()

# with open('/scratch/mfedor/final_conf_early_stopping.txt', 'w') as output_file:
print(f"recall: {recall}\tspecificity: {specificity}\tprecision: {precision}\tf1_score: {f1_score}\taccuracy: {accuracy}")
