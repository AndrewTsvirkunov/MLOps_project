"""
Программа: Модель для предсказания наземной концентрации оксида азота (4)
Версия: 1.0
"""

import warnings
import optuna
import pandas
import pandas as pd
import yaml

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.transform.transform import check_columns_evaluate
from src.pipeline.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = '../config/params.yml'

class Specifications(BaseModel):
    """
    Схема (проверка типов, валидация)
    Признаки для получения результатов модели
    """

    ID: str
    LAT: float
    LON: float
    Precipitation: float
    LST: float
    AAI: float
    CloudFraction: float
    NO2_strat: float
    TropopausePressure: float
    month: int
    year: int
    NO2_ratio: float
    Sum_Concentration: float


@app.get('/hello')
def welcome():
    """
    Hello
    :return: None
    """
    return {'message': 'Hello!'}


@app.post('/train')
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)
    return {'metrics': metrics}


@app.post('/predict')
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), 'Результат не соответствует типу list'
    return {'prediction': result[:5]}


@app.post('/predict_input')
def prediction_input(specifications: Specifications):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            specifications.ID,
            specifications.LAT,
            specifications.LON,
            specifications.Precipitation,
            specifications.LST,
            specifications.AAI,
            specifications.CloudFraction,
            specifications.NO2_strat,
            specifications.TropopausePressure,
            specifications.month,
            specifications.year,
            specifications.NO2_ratio,
            specifications.Sum_Concentration,
        ]
    ]

    cols = [
        'ID',
        'LAT',
        'LON',
        'Precipitation',
        'LST',
        'AAI',
        'CloudFraction',
        'NO2_strat',
        'TropopausePressure',
        'month',
        'year',
        'NO2_ratio',
        'Sum_Concentration'
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data, flg_input=True)[0]

    result = (
        {'High concentration'}
        if predictions >= 30.0
        else {'Low concentration'}
        if predictions < 30.0
        else 'Error result'
    )
    return result


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)

