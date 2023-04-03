import multiprocessing
import numpy as np
import tensorflow as tf
import time

from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.initializers import GlorotUniform, Zeros
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential, load_model
from math import floor


from .data_loader import DataLoader


SEED = 13

class SarsCoV2Classifier:

    def __init__(self,
        train_data_path: str,
        signal_length: int,
        signal_begin: int,
        train_batch_size: int,
        validation_batch_size: int
    ) -> None:
        self.train_data_path = train_data_path
        self.signal_length = signal_length
        self.signal_begin = signal_begin
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size

        self.train_data_loader = None
        self.validation_data_loader = None

        self.classifier = None

        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0


    def initialize_training(
        self,
        strides: int,
        kernel_size: int,
        use_data_factor: float=1,
        validation_data_factor: float=0.2,
        shuffle: bool=True
    ) -> None:
        train_data = np.load(self.train_data_path, allow_pickle=True, mmap_mode='r')
        train_data = train_data[:, self.signal_begin : train_data.shape[1]]

        validation_data_size = floor(train_data.shape[0] * validation_data_factor)
        validation_data = train_data[0 : validation_data_size]
        train_data = train_data[validation_data_size : train_data.shape[0]]

        self.train_data_loader = DataLoader(
            data=train_data,
            batch_size=self.train_batch_size,
            item_length=self.signal_length,
            use_items_factor=use_data_factor,
            shuffle=shuffle
        )

        self.validation_data_loader = DataLoader(
            data=validation_data,
            batch_size=self.validation_batch_size,
            item_length=self.signal_length,
            use_items_factor=1,
            shuffle=shuffle
        )

        input_shape = [len(train_data[0]) - 1, 1]

        self.classifier = Sequential()

        self.classifier.add(
            Conv1D(filters=64,
                kernel_size=kernel_size,
                strides=strides,
                activation='relu',
                padding='same',
                input_shape=input_shape,
                kernel_initializer=GlorotUniform(SEED),
                bias_initializer=Zeros()
            )
        )

        self.classifier.add(Dropout(rate=0.1, seed=SEED))

        self.classifier.add(
            Conv1D(filters=128,
            kernel_size=kernel_size,
            strides=strides,
            activation='relu',
            padding='same',
            kernel_initializer=GlorotUniform(SEED),
            bias_initializer=Zeros()
            )
        )

        self.classifier.add(MaxPooling1D(pool_size=2))

        self.classifier.add(Dropout(rate=0.1, seed=SEED))

        self.classifier.add(
            Conv1D(filters=128,
            kernel_size=kernel_size,
            strides=strides,
            activation='relu',
            padding='same',
            kernel_initializer=GlorotUniform(SEED),
            bias_initializer=Zeros()
            )
        )

        self.classifier.add(Flatten())
        self.classifier.add(Dense(2, activation='softmax', kernel_initializer=GlorotUniform(seed=13), bias_initializer=Zeros()))

        self.classifier.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001),  metrics=['accuracy'])


    def train(self, epochs: int) -> None:
        callback = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            verbose=True,
            mode='max',
            restore_best_weights=True
        )

        self.classifier.fit(
            x=self.train_data_loader,
            validation_data=self.validation_data_loader,
            epochs=epochs,
            verbose=True,
            steps_per_epoch=len(self.train_data_loader),
            callbacks=(callback,),
            workers=0,
            use_multiprocessing=False
        )


    def evaluate(self) -> None:
        score = self.classifier.evaluate(self.validation_data_loader)
        print("\n\ntest loss: {} | test acc: {}".format(score[0], score[1]))

        evaluation_times = []

        for test_examples, test_classes in self.validation_data_loader:
            start = time.time_ns()
            results = self.predict(test_examples)
            end = time.time_ns()

            evaluation_times.append(end - start)

            assert results is not None
            self._evaluate_predictions(test_classes, results)

        mean_time = np.mean(evaluation_times)
        max_time = max(evaluation_times)
        min_time = min(evaluation_times)

        recall = self._get_recall()
        specificity = self._get_specificity()
        precision = self._get_precision()
        f1_score = self._get_f1_score()

        print("mean: %.2f\tmax: %.2f\tmin:%.2f" % (mean_time, max_time, min_time))
        print("recall: %f\tspcificity: %f\tprecision: %f\tf1_score: %f" %
            (recall, specificity, precision, f1_score)
        )

        self._reset_counters()
        return recall, specificity, precision, f1_score, score[1]


    def _evaluate_predictions(self, labels, predictions):
        for label, prediction in zip(labels, predictions):
            predict_label = not prediction[0] > prediction[1]
            label = 0 if label[0] == 1 else 1

            if predict_label == 1 and label == 1:
                self.true_positives += 1
            if predict_label == 0 and label == 0:
                self.true_negatives += 1
            if predict_label == 1 and label == 0:
                self.false_positives += 1
            if predict_label == 0 and label == 1:
                self.false_negatives += 1


    def _reset_counters(self) -> None:
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0


    def predict(self, inputs: np.ndarray):
        return self.classifier(inputs)


    def load(self, path: str) -> None:
        self.classifier = load_model(path)


    def save(self, path: str) -> None:
        self.classifier.save(path)


    def _get_recall(self) -> float:
        try:
            return self.true_positives / (self.true_positives + self.false_negatives)
        except ZeroDivisionError:
            return 0


    def _get_specificity(self) -> float:
        try:
            return self.true_negatives / (self.true_negatives + self.false_positives)
        except ZeroDivisionError:
            return 0


    def _get_precision(self) -> float:
        try:
            return self.true_positives / (self.true_positives + self.false_positives)
        except ZeroDivisionError:
            return 0


    def _get_f1_score(self) -> float:
        recall = self._get_recall()
        precision = self._get_precision()

        try:
            return (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0


    def summarize_keras_trainable_variables(self):
        s = sum(map(lambda x: x.sum(), self.classifier.get_weights()))
        print("summary of trainable variables: %.13f" % (s))
        return s
