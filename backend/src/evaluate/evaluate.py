"""
Программа: предсказание на новых данных
Версия: 1.0
"""

import os
import yaml
import joblib
import pandas as pd
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess, pipeline_preprocess_input


def pipeline_evaluate(
    config_path, dataset: pd.DataFrame = None,  data_path: str = None, flg_input: bool = False
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param config_path: путь до конфигурационного файла
    :param dataset: датасет
    :param data_path: путь до файла с данными
    :param flg_input: флаг для вводимых данных
    :return: предсказания
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config['preprocessing']
    train_config = config['train']

    if data_path:
        dataset = get_dataset(dataset_path=data_path)

    if flg_input:
        dataset = pipeline_preprocess_input(data=dataset, **preprocessing_config)
    else:
        dataset = pipeline_preprocess(data=dataset, **preprocessing_config)

    model = joblib.load(os.path.join(train_config['model_path']))
    prediction = model.predict(dataset).tolist()

    return prediction