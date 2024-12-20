"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

from io import BytesIO
import json
import streamlit as st
import requests
import pandas as pd


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результатов
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    :return: None
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    ID = st.sidebar.selectbox('ID станции', unique_df['ID'])
    LAT = st.sidebar.slider(
        'Широта',
        min_value=min(unique_df['LAT']),
        max_value=max(unique_df['LAT'])
    )
    LON = st.sidebar.slider(
        'Долгота',
        min_value=min(unique_df['LON']),
        max_value=max(unique_df['LON'])
    )
    Precipitation = st.sidebar.slider(
        'Осадки',
        min_value=min(unique_df['Precipitation']),
        max_value=max(unique_df['Precipitation'])
    )
    LST = st.sidebar.slider(
        'Среднесуточная температура',
        min_value=min(unique_df['LST']),
        max_value=max(unique_df['LST'])
    )
    AAI = st.sidebar.slider(
        'Аэрозольный индекс',
        min_value=min(unique_df['AAI']),
        max_value=max(unique_df['AAI'])
    )
    CloudFraction = st.sidebar.slider(
        'Облачность',
        min_value=min(unique_df['CloudFraction']),
        max_value=max(unique_df['CloudFraction'])
    )
    NO2_strat = st.sidebar.selectbox(
        'NO2_strat',
        unique_df['NO2_strat']
        # min_value=min(unique_df['NO2_strat']),
        # max_value=max(unique_df['NO2_strat'])
    )
    TropopausePressure = st.sidebar.slider(
        'Тропопаузное давление',
        min_value=min(unique_df['TropopausePressure']),
        max_value=max(unique_df['TropopausePressure'])
    )
    month = st.sidebar.slider(
        'month', min_value=min(unique_df['month']), max_value=max(unique_df['month'])
    )
    year = st.sidebar.slider(
        'year', min_value=min(unique_df['year']), max_value=max(unique_df['year'])
    )
    NO2_ratio = st.sidebar.slider(
        'NO2_ratio',
        min_value=min(unique_df['NO2_ratio']),
        max_value=max(unique_df['NO2_ratio'])
    )
    Sum_Concentration = st.sidebar.selectbox(
        'Суммарная концентрация',
        unique_df['Sum_Concentration']
    )

    dict_data = {
        'ID': ID,
        'LAT': LAT,
        'LON': LON,
        'Precipitation': Precipitation,
        'LST': LST,
        'AAI': AAI,
        'CloudFraction': CloudFraction,
        'NO2_strat': NO2_strat,
        'TropopausePressure': TropopausePressure,
        'month': month,
        'year': year,
        'NO2_ratio': NO2_ratio,
        'Sum_Concentration': Sum_Concentration
    }

    st.write(
        f'''### Данные:\n
    1) ID станции: {dict_data['ID']}
    2) Широта: {dict_data['LAT']}
    3) Долгота: {dict_data['LON']}
    4) Осадки: {dict_data['Precipitation']}
    5) Среднесуточная температура: {dict_data['LST']}
    6) Аэрозольный индекс: {dict_data['AAI']}
    7) Облачность: {dict_data['CloudFraction']}
    8) NO2_strat: {dict_data['NO2_strat']}
    9) Тропопаузное давление: {dict_data['TropopausePressure']}
    10) month: {dict_data['month']}
    11) year: {dict_data['year']}
    12) NO2_ratio: {dict_data['NO2_ratio']}
    13) Суммарная концентрация: {dict_data['Sum_Concentration']}
    '''
    )

    button_ok = st.button('Predict')
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f'## {output[0]}')
        st.success('Success!')


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files: формат файла
    """
    button_ok = st.button('Predict')
    if button_ok:
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_['predict'] = output.json()['prediction']
        st.write(data_.head())