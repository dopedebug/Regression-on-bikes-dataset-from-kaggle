import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from dict_to import dict1



df = pd.DataFrame(pickle.load(open("bikes.pkl","rb")))

def bike_pred(name,year,ex_showroom_price,df = df):
    lr = LinearRegression()
    x  = df[df.columns.drop('selling_price')].values
    y = df['selling_price'].values
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=2)
    lr.fit(x_train,y_train)
    return lr.predict([[name,year,ex_showroom_price]])[0]



model = st.selectbox(
    'Choose the model of the bike:',
    [x for x in dict1.keys()])


year = st.select_slider(
    'Choose the year sold:',
    options= [1995, 1993, 2003, 1991, 1998, 1999, 1988, 2001, 1997, 2005, 2002, 2000, 2007, 2004, 2014, 2009, 2013, 2006, 2012, 2020, 2016, 2008, 2010, 2011, 2015, 2018, 2017, 2019])
st.write('Choosen year is ', year)

showroom_price = st.select_slider(
    "Select ex-showroom-price:",
    options= range(30490,1278000))
st.write("Selected amount:", showroom_price)

if st.button('predict'):
    st.write(bike_pred(dict1[model],year,showroom_price))






