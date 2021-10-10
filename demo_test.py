import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv('housing.csv')

if st.button('Отобразить первые пять строк'):
    st.write(df.head())

if st.button('Обучить модель'):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=0.2,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))

options = (range(10, 36, 5))
option = st.selectbox('задайте размер выборки:', options)
size = (int(option)/100)
st.write('размер выборки:', option,'%')

X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size = size,
                                                        random_state=2100)
regr_model = XGBRegressor()
regr_model.fit(X_train, y_train)
pred = regr_model.predict(X_test)
#st.write(pred)
df_new = pd.DataFrame(zip(y_test,pred), columns = ['Реальные значения', 'Предсказанные значения'])
st.write(df_new)
st.line_chart(df_new)