import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor


@st.cache
def preprocess_data(data_in):
    '''
    Кодирование категориальных признаков и определение целевого признака
    '''
    LE = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = LE.fit_transform(data[col])
    target = "Customer_care_calls"
    data[target] = data[target].astype("float")
    xArray = data.drop(target, axis=1)
    yArray = data[target]
    return xArray, yArray

data_load_state = st.text('Загрузка данных...')
data = pd.read_csv('Train.csv')
data_load_state.text('Данные загружены!')

st.subheader('Первые 5 значений')
st.write(data.head())

data_len = data.shape[0]

if st.checkbox('Показать корреляционную матрицу'):
    fig, ax = plt.subplots(figsize=(15,9))
    sns.heatmap(data.corr(method="pearson"), ax=ax,annot=True, fmt=".2f")
    st.pyplot(fig)

cv_slider = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=20, value=5, step=1)

rows_in_one_fold = int(data_len / cv_slider)
allowed_knn = int(rows_in_one_fold * (cv_slider-1))
st.write('Количество строк в наборе данных - {}'.format(data_len))
st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))


cv_knn = st.sidebar.slider('Количество ближайших соседей:', min_value=1, max_value=allowed_knn, value=5, step=1)

xData, yData = preprocess_data(data)

scores = cross_val_score(KNeighborsRegressor(n_neighbors=cv_knn), 
        xData, yData, scoring='r2', cv=cv_slider)


st.subheader('Оценка качества модели')
st.write('Значения R2 для отдельных фолдов')
st.line_chart(pd.DataFrame({"scores": scores}))
st.write('Максимальное значение R2 по всем фолдам:')
st.write("{}".format(np.amax(scores)))
st.write('Номер фолда наилучшего качества: {}'.format(np.argmax(scores)))