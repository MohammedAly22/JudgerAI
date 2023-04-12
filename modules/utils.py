from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer


def train_model(
    model_building_func: Callable[[], keras.models.Sequential],
    X_train_vectors: pd.DataFrame | np.ndarray | tf.Tensor,
    y_train: pd.Series,
    k: int = 4,
    num_epochs: int = 30,
    batch_size: int = 64,
) -> Tuple[
    List[keras.models.Sequential],
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
]:
    """
    Trains a model on `X_train_vectors` and `y_train` using k-fold cross-validation.

    Parameters:
    -----------
    - model_building_func : Callable[[], tf.keras.models.Sequential]
        A function that builds and compiles a Keras Sequential model.
    - X_train_vectors : pd.DataFrame
        The training input data.
    - y_train : pd.Series
        The training target data.
    - k : int, optional
        The number of folds for cross-validation (default is 4).
    - num_epochs : int, optional
        The number of epochs to train for (default is 30).
    - batch_size : int, optional
        The batch size to use during training (default is 64).

    Returns:
    --------
    A tuple containing the following items:
        - all_models : List[keras.models.Sequential]
            A list of `k` trained models.
        - all_losses : List[List[float]]
            A `k` by `num_epochs` list containing the training losses for each fold.
        - all_val_losses : List[List[float]]
            A `k` by `num_epochs` list containing the validation losses for each fold.
        - all_acc : List[List[float]]
            A `k` by `num_epochs` list containing the training accuracies for each fold.
        - all_val_acc : List[List[float]]
            A `k` by `num_epochs` list containing the validation accuracies for each fold.
    """

    num_validation_samples = len(X_train_vectors) // k

    all_models = []
    all_losses = []
    all_val_losses = []
    all_accuracies = []
    all_val_accuracies = []

    for fold in range(k):
        print(f"fold: {fold+1}")
        validation_data = X_train_vectors[
            num_validation_samples * fold: num_validation_samples * (fold + 1)
        ]
        validation_targets = y_train[
            num_validation_samples * fold: num_validation_samples * (fold + 1)
        ]

        training_data = np.concatenate(
            [
                X_train_vectors[: num_validation_samples * fold],
                X_train_vectors[num_validation_samples * (fold + 1):],
            ]
        )
        training_targets = np.concatenate(
            [
                y_train[: num_validation_samples * fold],
                y_train[num_validation_samples * (fold + 1):],
            ]
        )

        model = model_building_func()
        history = model.fit(
            training_data,
            training_targets,
            validation_data=(validation_data, validation_targets),
            epochs=num_epochs,
            batch_size=batch_size,
        )

        all_models.append(model)
        all_losses.append(history.history["loss"])
        all_val_losses.append(history.history["val_loss"])
        all_accuracies.append(history.history["accuracy"])
        all_val_accuracies.append(history.history["val_accuracy"])

    return (all_models, all_losses, all_val_losses, all_accuracies, all_val_accuracies)


def print_testing_loss_accuracy(
    all_models: List[keras.models.Sequential],
    X_test_vectors: pd.DataFrame | np.ndarray | tf.Tensor,
    y_test: pd.Series,
) -> None:
    """
    Displaying testing loss and testing accuracy of each model in `all_models`,
    and displaying their average.

    Parameters:
    ------------
    - all_models : List[keras.models.Sequential]
        A list of size `k` contains trained models.
    - X_test_vectors : pd.DataFrame
        Contains testing vectors.
    - y_test : pd.Series
        Contains testing labels.
    """

    sum_testing_losses = 0.0
    sum_testing_accuracies = 0.0

    for i, model in enumerate(all_models):
        print(f"model: {i+1}")
        loss_accuracy = model.evaluate(X_test_vectors, y_test, verbose=1)
        sum_testing_losses += loss_accuracy[0]
        sum_testing_accuracies += loss_accuracy[1]
        print("====" * 20)

    num_models = len(all_models)
    avg_testing_loss = sum_testing_losses / num_models
    avg_testing_acc = sum_testing_accuracies / num_models
    print(f"average testing loss: {avg_testing_loss:.3f}")
    print(f"average testing accuracy: {avg_testing_acc:.3f}")


def calculate_average_measures(
    all_losses: list[list[float]],
    all_val_losses: list[list[float]],
    all_accuracies: list[list[float]],
    all_val_accuracies: list[list[float]],
) -> Tuple[
    List[keras.models.Sequential],
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
]:
    """
    Calculate the average measures of cross-validated results.

    Parameters:
    ------------
    - all_losses : List[List[float]]
        A `k` by `num_epochs` list contains the values of training losses.
    - all_val_losses : List[List[float]]
        A `k` by `num_epochs` list contains the values of validation losses.
    - all_accuracies : List[List[float]]
        A `k` by `num_epochs` list contains the values of training accuracies.
    - all_val_accuracies : List[List[float]]
        A `k` by `num_epochs` list contains the values of validation accuracies.

    Returns:
    --------
    A tuple containing the following items:
        - avg_loss_hist : List[float]
            A list of length `num_epochs` contains the average of training losses.
        - avg_val_loss_hist : List[float]
            A list of length `num_epochs` contains the average of validaton losses.
        - avg_acc_hist : List[float]
            A list of length `num_epochs` contains the average of training accuracies.
        - avg_val_acc_hist : List[float]
            A list of length `num_epochs` contains the average of validation accuracies.
    """

    num_epochs = len(all_losses[0])
    avg_loss_hist = [np.mean([x[i] for x in all_losses])
                     for i in range(num_epochs)]
    avg_val_loss_hist = [
        np.mean([x[i] for x in all_val_losses]) for i in range(num_epochs)
    ]
    avg_acc_hist = [np.mean([x[i] for x in all_accuracies])
                    for i in range(num_epochs)]
    avg_val_acc_hist = [
        np.mean([x[i] for x in all_val_accuracies]) for i in range(num_epochs)
    ]

    return (avg_loss_hist, avg_val_loss_hist, avg_acc_hist, avg_val_acc_hist)


class Doc2VecModel:
    """Responsible of creating, initializing, and training Doc2Vec embeddings model."""

    def __init__(self, vector_size=50, min_count=2, epochs=100, dm=1, window=5) -> None:
        """
        Initalize a Doc2Vec model.

        Parameters:
        ------------
        - vector_size : int, optional
            Dimensionality of the feature vectors (Default is 50).
        - min_count : int, optional
            Ignores all words with total frequency lower than this (Default is 2).
        - epochs : int, optional
            Represents the number of training epochs (Default is 100).
        - dm : int, optional
            Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed (Default is 1).
        - window : int, optional
            The maximum distance between the current and predicted word within a
            sentence (Default is 5).
        """

        self.doc2vec_model = Doc2Vec(
            vector_size=vector_size,
            min_count=min_count,
            epochs=epochs,
            dm=dm,
            seed=865,
            window=window,
        )

    def train_doc2vec_embeddings_model(
        self, tagged_docs_train: List[TaggedDocument]
    ) -> Doc2Vec:
        """
        Train Doc2Vec model on `tagged_docs_train`.

        Parameters:
        ------------
        - tagged_docs_train : list[TaggedDocument]
            Contains the required format of training Doc2Vec model.

        Returns:
        --------
        - doc2vec_model : Doc2Vec
            The trained Doc2Vec model.
        """

        self.doc2vec_model.build_vocab(tagged_docs_train)
        self.doc2vec_model.train(
            tagged_docs_train,
            total_examples=self.doc2vec_model.corpus_count,
            epochs=self.doc2vec_model.epochs,
        )

        return self.doc2vec_model


class GloveModel:
    """Responsible for creating and generating the glove embedding layer"""

    def __init__(self) -> None:
        pass

    def _generate_glove_embedding_index(
            self,
            glove_file_path: str = 'GloVe/glove.6B.50d.txt') -> dict:
        """
        Responsible for generating glove embedding index.

        Parameters:
        ------------
        - glove_file_path : str
            Defines the path of the pretrained GloVe embeddings text file
            (Default is "GloVe/glove.6B.50d.txt").

        Returns:
        --------
        - embedding_index : dict
            Contains each word as a key, and its co-effeicents as a value.
        """

        embeddings_index = {}
        with open(glove_file_path, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        return embeddings_index

    def _generate_glove_embedding_matrix(
            self,
            word_index: dict,
            embedding_index: dict,
            max_length: int) -> np.ndarray:
        """
        Generating embedding matrix of each word in `word_index`.

        Parameters:
        -----------
        - word_index : dict
            Contains words as keys with there indicies as values.
        - embedding_index : dict
            Contains each word as a key, and its co-effeicents as a value.
        - max_length : int
            Defines the size of the embedding vector of each word in the
            embedding matrix.

        Returns:
        --------
        - embedding_matrix : np.ndarray
            Contains all embedding vectors for each word in`word_index`.
        """

        embedding_matrix = np.zeros((len(word_index) + 1, max_length))

        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def generate_glove_embedding_layer(
            self,
            glove_tokenizer: Tokenizer,
            max_length: int = 50) -> keras.layers.Embedding:
        """
        Create GloVe embedding layer for later usage in the neural network.

        Paramters:
        ----------
        - glove_tokenizer : Tokenizer
            Trained tokenizer on training data to extract word index from it.
        - max_length : int, optional
            Defines the maximum length of the output embedding vector for
            each word. (Default is 50).

        Returns:
        --------
        - embedding_layer : keras.layers.Embedding
            An embedding layer of size `word index + 1` by `max_length` with
            trained weights that can be used a vectorizer of case facts.
        """

        word_index = glove_tokenizer.word_index

        embedding_index = self._generate_glove_embedding_index()
        embedding_matrix = self._generate_glove_embedding_matrix(
            word_index, embedding_index, max_length)

        embedding_layer = keras.layers.Embedding(len(word_index) + 1,
                                                 max_length,
                                                 weights=[embedding_matrix],
                                                 input_length=max_length,
                                                 trainable=False)

        return embedding_layer
