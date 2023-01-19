import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np


st.set_page_config(page_title=" Cluster", page_icon=":tada:", layout="wide")
def create_page(content, page_title=""):
    st.title(page_title)
    st.write(content)
st.header("Klaster Kemiripan jarak dan kecepatan")
st.write("Oleh Kelompok\n"+
"- Azzura Hudzaifa Harun (2055301020)\n - Resti Yusfarima (2055301118)\n - Simon Rikki Purba (2055301158)")

def create_page(content):
    st.write(content)

content = """Bagian ini berfungsi untuk melakukan klaster terhadap kemiripan data jarak dan kecepatan
        Pada aplikasi ini kami menggunakan algoritma KMeans

        """
create_page(content)
st.write("---")

uploaded_file = st.file_uploader("File csv", type=["csv"])
if uploaded_file:
                driver = pd.read_csv(uploaded_file)
                driver = driver.drop(["linha", "car_or_bus","rating_weather", "rating_bus","rating","time"], axis = 1)
                st.dataframe(driver)

                # --- Menentukan variabel yang akan di klusterkan ---
                driver_x = driver.iloc[:, 1:3]
                # --- Memvisualkan persebaran data ---
                st.write("---")
                st.success("Visualisasi Dataset dengan ScatterPlot")
                fig, ax = plt.subplots(figsize=(5,3))
                ax.scatter(driver.distance, driver.speed, s =4, c = "c", marker = "o", alpha = 1)
                st.pyplot(fig)
                #--- Mengubah Variabel Data Frame Menjadi Array ---
                x_array =  np.array(driver_x)
                # --- Menstandarkan Ukuran Variabel ---
                scaler = MinMaxScaler()
                x_scaled = scaler.fit_transform(x_array)
                #--- Menentukan dan mengkonfigurasi fungsi kmeans ---
                kmeans = KMeans(n_clusters = 3, random_state=123)
                #--- Menentukan kluster dari data ---
                kmeans.fit(x_scaled)
                #--- Menambahkan Kolom "kluster" Dalam Data Frame Driver ---
                driver["kluster"] = kmeans.labels_
                st.write("---")
                st.title("Hasil Klustering K-Means")
                fig, ax = plt.subplots(figsize=(20,10))
                output = ax.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = driver.kluster, marker = "o", alpha = 1, )
                centers = kmeans.cluster_centers_
                ax.scatter(centers[:,0], centers[:,1], c='r', s=400, alpha=1 , marker="s")
                ax.set_xticks(np.arange(0,1, step=0.05))
                ax.set_yticks(np.arange(0,1, step=0.040))
                plt.colorbar (output)
                st.pyplot(fig)
                # output dalam dataset
                st.success("Hasil Klaster dalam Dataset")
                st.write(driver)
                st.error("Kesimpulan")
                st.write("Data yang memiliki karakteristik jarak dan kecepatan yang sama atau mirip akan dikelompokkan dalam satu kluster")









