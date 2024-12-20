"""
Программа: получение метрик
Версия: 1.0
"""

import json
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
import pandas as pd
import numpy as np


def create_dict_metrics(y_test: pd.Series, y_pred: pd.Series) -> dict:
    """
    Получение словаря с метриками для задачи регресии
    :param y_test: фактическое значение
    :param y_pred: предсказанное значение
    :return: словарь с метриками
    """
    dict_metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "MSE": round(mean_squared_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "WAPE": np.sum(np.abs(y_pred - y_test)) / np.sum(y_test) * 100,
    }
    return dict_metrics


def save_metrics(
    data_x: pd.DataFrame, data_y: pd.Series, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    :return:
    """
    result_metrics = create_dict_metrics(y_test=data_y, y_pred=model.predict(data_x))
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
