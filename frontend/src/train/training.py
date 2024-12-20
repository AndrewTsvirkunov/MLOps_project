"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_optimization_history


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    :return: None
    """
    if os.path.exists(config['train']['metrics_path']):
        with open(config['train']['metrics_path']) as json_file:
            old_metrics = json.load(json_file)
    else:
        old_metrics = {'MAE': 0, 'MSE': 0, 'RMSE': 0, 'WAPE': 0}

    with st.spinner('Модель подбирает параметры'):
        output = requests.post(endpoint, timeout=8000)
    st.success('Succes!')

    new_metrics = output.json()['metrics']

    MAE, MSE, RMSE, WAPE = st.columns(4)
    MAE.metric(
        'MAE',
        new_metrics['MAE'],
        f'{new_metrics["MAE"]-old_metrics["MAE"]:.4f}'
    )
    MSE.metric(
        'MSE',
        new_metrics['MSE'],
        f'{new_metrics["MSE"]-old_metrics["MSE"]:.4f}'
    )
    RMSE.metric(
        'RMSE',
        new_metrics['RMSE'],
        f'{new_metrics["RMSE"]-old_metrics["RMSE"]:.4f}'
    )
    WAPE.metric(
        'WAPE',
        new_metrics['WAPE'],
        f'{new_metrics["WAPE"]-old_metrics["WAPE"]:.4f}'
    )

    study = joblib.load(os.path.join(config['train']['study_path']))
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_history, use_container_width=True)

