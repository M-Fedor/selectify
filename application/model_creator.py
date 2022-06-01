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


session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(session)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

classifier = SarsCoV2Classifier(
    train_data_path='/scratch/mfedor/training_data.npy',
    test_data_path='/scratch/mfedor/validation_data.npy',
    signal_length=2000,
    train_batch_size=8192,
    test_batch_size=32
)

reset_random_generators()

classifier.initialize_training(strides=7, kernel_size=60, use_data_factor=1, shuffle=True)
classifier.summarize_keras_trainable_variables()
classifier.train(epochs=100)
recall, specificity, precision, f1_score, accuracy = classifier.evaluate()

with open('/scratch/mfedor/final_conf_early_stopping.txt', 'w') as output_file:
    print("recall: %f\tspecificity: %f\tprecision: %f\tf1_score: %f\taccuracy: %f" %
        (recall, specificity, precision, f1_score, accuracy), file=output_file)
