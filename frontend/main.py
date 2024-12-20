"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os
import yaml
import streamlit as st
import seaborn as sns
import geopandas as gp
from src.data.get_data import load_data, get_dataset, get_geodataset
from src.plotting.charts import kde_bar_plot, barplot, lineplot, get_bins, maps, data_geo
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

CONFIG_PATH = '../config/params.yml'


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        'https://avatars.dzeninfra.ru/get-zen_doc/3737694/pub_61efd762ac911e65410ae0bd_61efd79efaaefe16cd09f514/scale_1200',
        width=500)

    st.markdown(
        '# MLOps project\n'
        '## Concentration NO2 Prediction'
    )
    # st.title()
    st.write(
        '''
        https://zindi.africa/competitions/geoai-ground-level-no2-estimation-challenge\n
        Можно ли оценить концентрацию NO2 на поверхности земли, используя данные дистанционного зондирования?
        В рамках задания GeoAI Ground-level NO2 Evaluation Challenge необходимо разработать модели 
        машинного обучения для оценки концентраций NO2 на поверхности в различных погодных условиях и
        временах года, демонстрируя адаптивность и надежность.
        Мы имеем данные наземных измерений со станций мониторинга качества воздуха в континентальной части 
        итальянских регионов Ломбардия и Венето, а также данные дистанционного зондирования NO2 
        от Sentinel-5P TROPOMI, данные об осадках от CHIRPS и данные о температуре поверхности земли 
        от NOAA.\n
        Задача регрессии.
        '''
    )

    st.markdown(
        '''
        ### Данные
            - ID_Zindi - id наблюдения
            - ID - номер станции
            - Date - дата
            - LAT - широта
            - LON - долгота
            - Precipitation - осадки
            - LST - суточная температура поверхности суши
            - AAI - аэрозольный индекс
            - CloudFraction - эффективная доля облаков
            - NO2_strat - стратосферная концентрация оксида азота
            - NO2_total - общая концентрация оксида азота
            - NO2_trop - тропосферная концентрация оксида азота
            - TropopausePressure - давление в тропопаузе
            - GT_NO2 - поверхностная концентрация оксида азота (целевая переменная)
        ### Новые данные
            - month - месяц
            - year - год
            - NO2_ratio - отношение тропосферной концентрации к стратосферной
            - Sum Concentration - суммарная концентрация 
        '''
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown(
        '# Exploratory data analysis\n'
        '### Датасет:')

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data = get_dataset(dataset_path=config['preprocessing']['train_path'])
    prov_data = get_geodataset(geodataset_path=config['preprocessing']['geodata_path'])

    st.write(data.head())

    map_station = st.sidebar.checkbox('Координаты станций мониторинга')
    map_concentration = st.sidebar.checkbox('Наземная концентрация NO2 по станциям мониторинга')
    daily_temperature = st.sidebar.checkbox('Суточная температура поверхности')
    aerosol_index = st.sidebar.checkbox('Аэрозольный индекс')
    cloud_fraction = st.sidebar.checkbox('Эффективная доля облаков')
    tropopause_pressure = st.sidebar.checkbox('Давление в тропопаузе')
    NO2_strat = st.sidebar.checkbox('Концентрация NO2 в стратосфере')
    month_year = st.sidebar.checkbox('Динамика во времени')

    if map_station:
        geo_data = data_geo(data=data)

        st.pyplot(
            maps(
                prov_data=prov_data,
                geo_data=geo_data,
                title='Координаты станций мониторинга (Ломбардия и Венето)',
                color='darkred',
            )
        )


    if map_concentration:
        geo_data = data_geo(data=data)

        st.pyplot(
            maps(
                prov_data=prov_data,
                geo_data=geo_data,
                title='Наземная концентрация NO2 по станциям мониторинга',
                column='GT_NO2',
                marker='D',
                markersize=100,
                cmap='hot'
            )
        )
        st.write("Чем темнее оттенок, тем ниже концентрация (по шкале cmap='hot')")

    if daily_temperature:
        data = get_bins(data=data,
                        col='LST',
                        name=config['preprocessing']['bins_columns'][0])
        st.pyplot(
            kde_bar_plot(
                data=data,
                x=config['preprocessing']['bins_columns'][0],
                y=config['preprocessing']['target_column'],
                palette='afmhot_r',
                title='Суточная температура поверхности'
            )
        )
        st.write('Чем выше температура на поверхности, тем меньше концентрация оксида азота.')

    if aerosol_index:
        data = get_bins(data=data,
                        col='AAI',
                        name=config['preprocessing']['bins_columns'][1])
        st.pyplot(
            kde_bar_plot(
                data=data,
                x=config['preprocessing']['bins_columns'][1],
                y=config['preprocessing']['target_column'],
                palette='Dark2',
                title='Аэрозольный индекс'
            )
        )
        st.write('При самом низком аэрозольном индексе содержание NO2 наименьшее.')

    if cloud_fraction:
        data = get_bins(data=data,
                        col='CloudFraction',
                        name=config['preprocessing']['bins_columns'][2])
        st.pyplot(
            kde_bar_plot(
            data=data,
            x=config['preprocessing']['bins_columns'][2],
            y=config['preprocessing']['target_column'],
            palette='cool',
            title='Эффективная доля облаков'
            )
        )
        st.write('Cначала можно увидеть, что чем меньше облачность, тем ниже концентрация оксида азота,'
                 'но далее последовательность нарушается.')
        st.write('Концентрация оксида не зависит от доли доли облаков.')

    if tropopause_pressure:
        data = get_bins(data=data,
                        col='TropopausePressure',
                        name=config['preprocessing']['bins_columns'][3])
        st.pyplot(
            kde_bar_plot(
                data=data,
                x=config['preprocessing']['bins_columns'][3],
                y=config['preprocessing']['target_column'],
                palette='rocket',
                title='Давление в тропопаузе'
            )
        )
        st.write('Чем меньше давление в тропопаузе, тем ниже концентрация NO2.')

    if NO2_strat:
        data = get_bins(data=data,
                        col='NO2_strat',
                        name=config['preprocessing']['bins_columns'][4])
        st.pyplot(
            barplot(
                data=data,
                x=config['preprocessing']['target_column'],
                y=config['preprocessing']['bins_columns'][4],
                palette='ocean',
                title='Концентрация NO2 в стратосфере'
            )
        )
        st.write('Поверхностная концентрация оксида азота обратнопропорциональна стратосферной.')

    if month_year:
        st.pyplot(
            lineplot(
                data=data,
                x=config['preprocessing']['drop_columns'][1],
                y=config['preprocessing']['target_column'],
                title='Динамика во времени'
            )
        )
        st.write('При более высокой температуре концентрация оксида азота ниже, '
                 'следовательно в зимние месяцы концентрация должна быть выше.')



def training():
    """
    Тренировка модели
    """
    st.markdown('# Training model LightGBM')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['train']

    if st.button('Start training'):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown('# Prediction')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_input']
    unique_data_path = config['preprocessing']['unique_values_path']

    # проверка на наличие сохраненной модели
    if os.path.exists(config['train']['model_path']):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error('Сначала обучите модель')


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown('# Prediction')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_from_file']

    upload_file = st.file_uploader(
        'Выберите файл:', type=['csv', 'xlsx'], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data='Test')
        # проверка на наличие сохраненной модели
        if os.path.exists(config['train']['model_path']):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error('Сначала обучите модель')

def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        'Описание проекта': main_page,
        'Exploratory data analysis': exploratory,
        'Training model': training,
        'Prediction': prediction,
        'Prediction from file': prediction_from_file,
    }
    selected_page = st.sidebar.selectbox('Выберите пункт', page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == '__main__':
    main()