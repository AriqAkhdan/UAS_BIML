import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

#https://www.youtube.com/watch?v=8M20LyCZDOY
st.write("""
# Web Prediksi Jenis Pinguin

Web ini memprediksi jenis dari spesies Pinguin Palmer berdasarkan input data yang diberikan.

Data yang digunakan untuk membentuk model prediksi didapatkan dari [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) di R yang disediakan oleh Allison Horst.
""")

st.sidebar.header('Input Data')

# Memasukkan input data ke dalam dataframe
def user_input_features():
    island = st.sidebar.selectbox('Pulau',('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Jenis Kelamin',('male','female'))
    bill_length_mm = st.sidebar.slider('Panjang Paruh (mm)', 32.1,59.6,43.9)
    bill_depth_mm = st.sidebar.slider('Kedalaman Paruh (mm)', 13.1,21.5,17.2)
    flipper_length_mm = st.sidebar.slider('Panjang Sirip (mm)', 172.0,231.0,201.0)
    body_mass_g = st.sidebar.slider('Massa Tubuh (g)', 2700.0,6300.0,4207.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Menggabungkan input data dengan dataset pinguin
penguins_raw = pd.read_csv('https://raw.githubusercontent.com/AriqAkhdan/UAS_BIML/main/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'], axis=1)
df = pd.concat([input_df,penguins],axis=0)

# Encoding variabel ordinal
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]

# Menampilkan fitur input data
st.subheader('Input Data')

st.write(df)

# Memuat model klasifikasi
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Menerapkan model untuk membuat prediksi
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediksi Jenis Pinguin')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediksi Probabilitas Jenis Pinguin')
st.write(prediction_proba)
st.write("""
Keterangan:
0 = Adelie;
1 = Chinstrap;
2 = Gentoo
""")
