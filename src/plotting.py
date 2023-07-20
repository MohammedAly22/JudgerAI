from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow import keras


class PlottingManager:
    """Responsible for providing plots & visualization for the models."""

    def __init__(self) -> None:
        """Define style for visualizations."""
        plt.style.use("seaborn")

    def plot_subplots_curve(
        self,
        training_measure: List[List[float]],
        validation_measure: List[List[float]],
        title: str,
        train_color: str = "orangered",
        validation_color: str = "dodgerblue",
    ) -> None:
        """
        Plotting subplots of the elements of `training_measure` vs. `validation_measure`.

        Parameters:
        ------------
        - training_measure : List[List[float]]
            A `k` by `num_epochs` list contains the trained measure whether it's loss or
            accuracy for each fold.
        - validation_measure : List[List[float]]
            A `k` by `num_epochs` list contains the validation measure whether it's loss
            or accuracy for each fold.
        - title : str
            Represents the title of the plot.
        - train_color : str, optional
            Represents the graph color for the `training_measure`. (Default is "orangered").
        - validation_color : str, optional
            Represents the graph color for the `validation_measure`. (Default is "dodgerblue").
        """

        plt.figure(figsize=(12, 8))

        for i in range(len(training_measure)):
            plt.subplot(2, 2, i + 1)
            plt.plot(training_measure[i], c=train_color)
            plt.plot(validation_measure[i], c=validation_color)
            plt.title("Fold " + str(i + 1))

        plt.suptitle(title)
        plt.show()

    def plot_heatmap(
        self, measure: List[List[float]], title: str, cmap: str = "coolwarm"
    ) -> None:
        """
        Plotting a heatmap of the values in `measure`.

        Parameters:
        ------------
        - measure : List[List[float]]
            A `k` by `num_epochs` list contains the measure whether it's loss
            or accuracy for each fold.
        - title : str
            Title of the plot.
        - cmap : str, optional
            Color map of the plot (default is "coolwarm").
        """

        # transpose the array to make it `num_epochs` by `k`
        values_array = np.array(measure).T
        df_cm = pd.DataFrame(
            values_array,
            range(1, values_array.shape[0] + 1),
            ["fold " + str(i + 1) for i in range(4)],
        )

        plt.figure(figsize=(10, 8))
        plt.title(
            title + " Throughout " + str(values_array.shape[1]) + " Folds", pad=20
        )
        sn.heatmap(df_cm, annot=True, cmap=cmap, annot_kws={"size": 10})
        plt.show()

    def plot_average_curves(
        self,
        title: str,
        x: List[float],
        y: List[float],
        x_label: str,
        y_label: str,
        train_color: str = "orangered",
        validation_color: str = "dodgerblue",
    ) -> None:
        """
        Plotting the curves of `x` against `y`, where x and y are training and validation
        measures (loss or accuracy).

        Parameters:
        ------------
        - title : str
            Title of the plot.
        - x : List[float]
            Training measure of the models (loss or accuracy).
        - y : List[float]
            Validation measure of the models (loss or accuracy).
        - x_label : str
            Label of the training measure to put it in plot legend.
        - y_label : str
            Label of the validation measure to put it in plot legend.
        - train_color : str, optional
            Color of the training plot (default is "orangered").
        - validation_color : str, optional
            Color of the validation plot (default is "dodgerblue").
        """

        plt.title(title, pad=20)
        plt.plot(x, c=train_color, label=x_label)
        plt.plot(y, c=validation_color, label=y_label)
        plt.legend()
        plt.show()

    def plot_roc_curve(
        self,
        all_models: List[keras.models.Sequential],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """
        Plotting the AUC-ROC curve of all the passed models in `all_models`.

        Parameters:
        ------------
        - all_models : List[keras.models.Sequential]
            Contains all trained models, number of models equals number of
             `k` fold cross-validation.
        - X_test : pd.DataFrame
            Contains the testing vectors.
        - y_test : pd.Series
            Contains the testing labels.
        """

        plt.figure(figsize=(12, 8))
        for i, model in enumerate(all_models):
            y_pred = model.predict(X_test).ravel()
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_curve = auc(fpr, tpr)
            plt.subplot(2, 2, i + 1)
            plt.plot([0, 1], [0, 1], color="dodgerblue", linestyle="--")
            plt.plot(
                fpr,
                tpr,
                color="orangered",
                label=f"Fold {str(i+1)} (area = {auc_curve:.3f})",
            )
            plt.legend(loc="best")
            plt.title(f"Fold {str(i+1)}")

        plt.suptitle("AUC-ROC curves")
        plt.show()

    def plot_classification_report(
        self, model: keras.models.Sequential, X_test: pd.DataFrame, y_test: pd.Series
    ) -> str | dict:
        """
        Plotting the classification report of the passed `model`.

        Parameters:
        ------------
        - model : keras.models.Sequential
            The trained model that will be evaluated.
        - X_test : pd.DataFrame
            Contains the testing vectors.
        - y_test : pd.Series
            Contains the testing labels.

        Returns:
        --------
        - str | dict: The classification report for the given model and testing data.
            It returns a string if `output_format` is set to 'str', and returns
            a dictionary if `output_format` is set to 'dict'.
        """

        y_pred = model.predict(X_test).ravel()
        preds = np.where(y_pred > 0.5, 1, 0)
        cls_report = classification_report(y_test, preds)

        return cls_report

    def plot_confusion_matrix(
        self,
        all_models: List[keras.models.Sequential],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """
        Plotting the confusion matrix of each model in `all_models`.

        Parameters:
        ------------
        - all_models: list[keras.models.Sequential]
            Contains all trained models, number of models equals
            number of `k` fold cross-validation.
        - X_test: pd.DataFrame
            Contains the testing vectors.
        - y_test: pd.Series
            Contains the testing labels.
        """

        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        for i, (model, ax) in enumerate(zip(all_models, axes.flatten())):
            y_pred = model.predict(X_test).ravel()
            preds = np.where(y_pred > 0.5, 1, 0)

            conf_matrix = confusion_matrix(y_test, preds)
            sn.heatmap(conf_matrix, annot=True, ax=ax)
            ax.set_title(f"Fold {i+1}")

        plt.suptitle("Confusion Matrices")
        plt.tight_layout()
        plt.show()
