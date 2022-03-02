from re import X
from statistics import linear_regression
import numpy as np
import pandas as pd
import streamlit as st



st.title("WINE QUALITY PREDICTOR")
option=st.selectbox('Which wine quality do u need to find ?' ,('White_wine','Red_wine'))





if option=='White_wine':
    df=pd.read_csv('winequality-white.csv',delimiter=';')
else:
    df=pd.read_csv('winequality-red.csv',delimiter=';')

x=df.drop('quality',axis=1)
y=df['quality']




from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.1,random_state=42)





from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)



from sklearn.metrics import mean_squared_error,r2_score

mse=mean_squared_error(y_predict,y_test)

accuracy=r2_score(y_test,y_predict)*100



st.header("Results :")

st.write("Mean Squared Error : ",mse)

st.write("Accuracy : " ,accuracy)



