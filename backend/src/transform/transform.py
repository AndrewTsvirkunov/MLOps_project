"""
Программа: предобработка тренировачных
и тестовых данных
Версия 1.0
"""

import json
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


def save_unique_for_train(
        data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """Сохранение словаря с признаками и уникальными значениями
    param: data: датасет
    param: drop_columns: список с признаками для удаления
    param: target_column: целевая переменная
    param: unique_values_path: путь до файла со словарем
    """
    unique_df = data.drop(
        columns=[drop_columns[0]] + [target_column], axis=1, errors='ignore')

    # создаем словарь с уникальными значениями для вывода
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, 'w') as file:
        json.dump(dict_unique, file)


def check_columns_evaluate(data: pd.DataFrame,
                           unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train
    и упорядочивание признаков согласно train
    param: data: test датасет
    param: unique_values_path: путь до списка с признаками из train
    return: test датасет
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), 'Разные признаки'
    return data[column_sequence]


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преоборазование признаков в разный тип данных
    param: data: датасет
    param: change_type_columns: словарь с признаками и типами данных
    """
    return data.astype(change_type_columns, errors='raise')


def feature_engineering(data: pd.DataFrame, **kwargs):
    """
    Feature engineering
    :param data: датасет
    :param kwargs: переменная
    :return: новые признаки
    """
    # разобьем столбец с датой на два: месяц и год
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year

    # создадим новый признак, который будет указывать на соотношение
    # тропосфреной и стратосферной концентраций
    data['NO2_ratio'] = data['NO2_trop'] / data['NO2_strat']

    # создадим признак суммарной концентрации
    data["Sum_Concentration"] = data["NO2_strat"] + data["NO2_total"] + data["NO2_trop"]

    # переведем градусы Кельвина в градусы Цельсия
    data['LST'] = data['LST'] - 273.15

    # удалим лишние признаки
    data = data.drop(kwargs['drop_columns'][1:], axis=1)

    return data


def fillna_data(data: pd.DataFrame, list_median: list, list_mean: list):
    """
    Функция заполнения пропусков разными значениями
    data: датасет
    list_median: список с признаками, необходимых заполнить медианой
    list_mean: список с признаками, необходимых заполнить средними
    """
    for m in list_median:
        data[m] = data[m].fillna(data[m].median())

    for n in list_mean:
        data[n] = data[n].fillna(data[n].mean())

    return data


def pipeline_preprocess(data: pd.DataFrame, flg_evaluate: bool = True, **kwargs):
    """
    params: data: датасет
    params: flg_evaluate: флаг для evaluate
    return: итоговый датасет
    """
    data = data.drop(kwargs['drop_columns'][0], axis=1, errors='ignore')
    data = data.drop(kwargs['bins_columns'], axis=1, errors='ignore')

    data = transform_types(data=data,
                           change_type_columns=kwargs['change_type_columns'])

    data = feature_engineering(data=data, **kwargs)

    data = fillna_data(data=data, list_median=kwargs['list_median'], list_mean=kwargs['list_mean'])

    if kwargs['target_column'] in data:
        data[kwargs['target_column']] = (data[kwargs['target_column']]
                                         .fillna(data[kwargs['target_column']].median()))

    dict_category = {key: 'category' for key in data.select_dtypes(['object']).columns}
    data = transform_types(data=data, change_type_columns=dict_category)

    if flg_evaluate:
        data = check_columns_evaluate(
            data=data, unique_values_path=kwargs['unique_values_path'])
    else:
        save_unique_for_train(
            data=data,
            drop_columns=kwargs['drop_columns'],
            target_column=kwargs['target_column'],
            unique_values_path=kwargs['unique_values_path'])

    return data


def pipeline_preprocess_input(data: pd.DataFrame, **kwargs):
    """
    Пайплайн для предсказаний по введенным значениям
    :param data: датасет
    :param kwargs: переменная
    :return: итоговый датасет
    """
    data = check_columns_evaluate(
        data=data, unique_values_path=kwargs['unique_values_path'])

    dict_category = {key: 'category' for key in data.select_dtypes(['object']).columns}
    data = transform_types(data=data, change_type_columns=dict_category)

    return data