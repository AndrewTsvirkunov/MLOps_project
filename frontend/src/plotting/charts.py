"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import geopandas as gp
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point
import seaborn as sns


def kde_bar_plot(
    data: pd.DataFrame, x: str, y:str, palette: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графиков kdeplot и barplot
    :param data: датасет
    :param x: признак для анализа
    :param y: целевая переменная
    :param palette: цвет рисунка
    :param title: название рисунка
    :return: рисунок
    """
    fig, axes = plt.subplots(nrows=2, figsize=(12, 12))

    sns.kdeplot(data=data, x=y, hue=x, palette=palette, ax=axes[0])

    axes[0].set_xlabel(y)
    axes[0].set_ylabel('Dentsity')

    sns.barplot(y=y, x=x, data=data, palette=palette, ax=axes[1])

    axes[1].set_xlabel(x.split('_')[0])
    axes[1].set_ylabel(y)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def barplot(
        data: pd.DataFrame, x: str, y: str, palette: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика barplot
    :param data: датасет
    :param x: целевая переменая
    :param y: признак
    :param palette: цвет рисунка
    :param title: название рисунка
    :return: рисунок
    """
    fig = plt.figure(figsize=(15, 7))
    sns.barplot(x=x,
                y=y,
                data=data,
                palette=palette)

    plt.title(title, fontsize=18)
    plt.xlabel(x, fontsize=14)
    plt.ylabel(y, fontsize=14)
    return fig


def lineplot(
        data: pd.DataFrame, x: str, y: str, title: str
) -> matplotlib.figure.Figure:
    """
    Отрисовка графика lineplot
    :param data: датасет
    :param x: признак
    :param y: целевая переменная
    :return: рисунок
    """
    fig = plt.figure(figsize=(12, 6))

    sns.lineplot(x=x, y=y, data=data)

    plt.xticks(rotation=45, fontsize=5)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    return fig


def get_bins(
    data: pd.DataFrame, col: str, name: str
) -> pd.DataFrame:
    """
    Функция разделения признака по квартилям
    для разведочного анализа распределения
    :param col: название столбца
    :param name: название нового бинаризованного столбца
    """
    bin_min = data[col].describe().loc['min']
    bin_25 = data[col].describe().loc['25%']
    bin_50 = data[col].describe().loc['50%']
    bin_75 = data[col].describe().loc['75%']
    bin_max = data[col].describe().loc['max']

    data[name] = pd.cut(data[col],
                      bins=[bin_min, bin_25, bin_50, bin_75, bin_max],
                      labels=['low', 'middle', 'middle+', 'high'])
    return data


def maps(
    prov_data: gp.GeoDataFrame, geo_data: gp.GeoDataFrame, title: str, color: str = None,
    column: str = None, marker: str = None, markersize: str = None, cmap: str = None
) -> matplotlib.figure.Figure:
    """
    Отрисовка карты Италии в Ломбардии и Венето
    :param prov_data: геодатасет с провинциями
    :param geo_data: геодатасет с исходными данными
    :param title: заголовок
    :return: карта
    """
    fig, ax = plt.subplots(1, figsize=(20, 10))
    base = prov_data[prov_data['DEN_REG'].isin(['Lombardia', 'Veneto']) == True].plot(ax=ax, color='green')
    geo_data.plot(ax=base, color=color, column=column, marker=marker, markersize=markersize, cmap=cmap)
    ax.set_title(title, fontsize=25)
    return fig

def data_geo(data: pd.DataFrame,) -> gp.GeoDataFrame:
    geo_data = data.copy()
    geo_data['coordinates'] = geo_data[['LON', 'LAT']].values.tolist()
    geo_data['coordinates'] = geo_data['coordinates'].apply(Point)
    geo_data = gp.GeoDataFrame(geo_data, geometry='coordinates')
    return geo_data
