"""
Программа: Получение данных из файла и преобразование
Версия: 1.0
"""

import io
from io import BytesIO
from typing import Dict, Tuple
import pandas as pd
import geopandas as gp
import streamlit as st


def get_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_csv(dataset_path)


def get_geodataset(geodataset_path: str) -> gp.GeoDataFrame:
    """
    Получение геоданных по заданному пути
    :param geodataset_path: путь до данных
    :return: геодатасет
    """
    return gp.read_file(geodataset_path).to_crs({'init':'epsg:4326'})


def load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data: данные
    :param type_data: тип датасета (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    dataset = pd.read_csv(data)
    st.write('Dataset load')
    st.write(dataset.head())

    dataset_bytes_obj = io.BytesIO()
    dataset.to_csv(dataset_bytes_obj, index=False)
    dataset_bytes_obj.seek(0)

    files = {
        'file': (f'{type_data}_dataset.csv', dataset_bytes_obj, 'multipart/form-data')
    }
    return dataset, files