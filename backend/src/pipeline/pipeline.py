"""Программа: Сборный конвейр для тренировки модели
Версия: 1.0
"""

import os
import joblib
import yaml

from ..data.split_dataset import split_train_test
from ..train.train import find_optimal_params, train_model
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели
    :param config_path: путь до конфигурационного файла
    :return: None
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']
    train_config = config['train']

    train_data = get_dataset(dataset_path=preprocessing_config['train_path'])

    df_train, df_test = split_train_test(dataset=train_data, **preprocessing_config)

    df_train = pipeline_preprocess(
        data=df_train, flg_evaluate=False, **preprocessing_config)

    df_test = pipeline_preprocess(
        data=df_test, flg_evaluate=False, **preprocessing_config)

    study = find_optimal_params(data_train=df_train, data_test=df_test, **train_config)

    lgbm = train_model(
        data_train=df_train,
        data_test=df_test,
        study=study,
        target=preprocessing_config['target_column'],
        metric_path=train_config['metrics_path']
    )

    joblib.dump(lgbm, os.path.join(train_config['model_path']))
    joblib.dump(study, os.path.join(train_config['study_path']))