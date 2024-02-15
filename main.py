#fichier principal pour faire fonctionner application
# pour demarrer streamlit : streamlit run main.py
# pour se placer dans environnement MPA : conda activate MPA
# pour generer environnement.yml : conda env export > environment.yml
# pour mettre en place sur un nouvel ordinateur environment MPA avec le fichier environement.yml : conda env create -f environment.yml
# pour ouvrir  anaconda navigotor ( gestionnaire environemment) : anaconda-navigator

from statistics import mean
import gsw
from matplotlib import pyplot as plt 
import streamlit as st
import xarray as xr
from argopy import DataFetcher as ArgoDataFetcher
import folium
from streamlit_folium import folium_static
import numpy as np
import pandas as pd
import datetime as dt
import numpy as np
from pyxpcm.models import pcm
import xarray as xr
from argopy import DataFetcher as ArgoDataFetcher
import math


def int_to_rgb(i):
    r = hex((i*i+10)%15).replace("0x","")
    g = hex((i*5)%15).replace("0x","")
    b = hex((i*i*i*2)%15).replace("0x","")
    return "#"+r*2+g*2+b*2

def recup_argo_data(llon:float,rlon:float, llat:float,ulat:float, depthmin:float, depthmax:float,intervalle:int,time_in:str,time_f:str):

     # Temporarily redefine np.int to int -> car sinon on a une erreur np.int est deprecier dans NumPy 1.20
    np.int = int

    #recuperer les données avec argopy
    ds_points = ArgoDataFetcher(src='erddap', parallel=True).region([llon,rlon, llat,ulat, depthmin, depthmax,time_in,time_f]).to_xarray()
    #mettre en 2 dimensions ( N_PROf x N_LEVELS)
    ds = ds_points.argo.point2profile()
    """"""
    #recuperer les données SIG0 et N2(BRV2) avec teos 10
    ds.argo.teos10(['SIG0','N2'])

    # z est créé pour représenter des profondeurs de la profondeur minimum à la profondeur maximale avec un intervalle de 5 mètres ( peux etre modifié)
    z=np.arange(depthmin,depthmax,intervalle)
    #interpole ds avec les profondeurs z
    ds2 = ds.argo.interp_std_levels(z)

    #Calculer la profondeur avec gsw.z_from_p a partir de la p (PRES) et de lat (LATITUDE)
    p=np.array(ds2.PRES)
    lat=np.array(ds2.LATITUDE)
    z=np.ones_like(p)
    nprof=np.array(ds2.N_PROF)

    for i in np.arange(0,len(nprof)):
        z[i,:]=gsw.z_from_p(p[i,:], lat[i])


    # Calcul de la profondeur à partir de la pression interpolée 
    p_interp=np.array(ds2.PRES_INTERPOLATED)
    z_interp=gsw.z_from_p(p_interp, 25) 


    #Créer un objet Dataset xarray pour stocker les données
    temp=np.array(ds2.TEMP)
    sal=np.array(ds2.PSAL)
    depth_var=z
    depth=z_interp
    lat=np.array(ds2.LATITUDE)
    lon=np.array(ds2.LONGITUDE)
    time=np.array(ds2.TIME)
    sig0 =np.array(ds2.SIG0)
    brv2 =np.array(ds2.N2)

    #ranger les données dans xarrays
    da=xr.Dataset(data_vars={
                            'TIME':(('N_PROF'),time),
                            'LATITUDE':(('N_PROF'),lat),
                            'LONGITUDE':(('N_PROF'),lon),
                            'TEMP':(('N_PROF','DEPTH'),temp),
                            'PSAL':(('N_PROF','DEPTH'),sal),
                            'SIG0':(('N_PROF','DEPTH'),sig0),
                            'BRV2':(('N_PROF','DEPTH'),brv2)
                            },
                            coords={'DEPTH':depth})
    np.int = np.int_
    return da

# pyxpcm profil salinité et température 
def pyxpcm_sal_temp(da, k ):

    #copy da pour éviter ecrasement de da
    ds_sal_temp = da.copy(deep=True)
    # redefinir temporairement np.int pour eviter erreur depreciation Numpy 1.20
    np.int = int
    element_depth = ds_sal_temp.sel(DEPTH=ds_sal_temp.DEPTH[-1]).DEPTH.values# récupère la valeur max de DEPTH
    z = np.arange(0.,element_depth,-10.) 
    pcm_features = {'temperature': z, 'salinity':z} 
    m = pcm(K=k, features=pcm_features, maxvar=2) # créer le model PCM
    
    features_zdim='DEPTH'
    features_in_ds = {'temperature': 'TEMP', 'salinity': 'PSAL'}
    #PCM propercies
    ds_pcm_sal_temp = ds_sal_temp.pyxpcm.fit_predict(m, features=features_in_ds, dim=features_zdim, inplace=True)
    ds_pcm_sal_temp['TEMP'].attrs['feature_name'] = 'temperature'
    ds_pcm_sal_temp['PSAL'].attrs['feature_name'] = 'salinity'
    ds_pcm_sal_temp['DEPTH'].attrs['axis'] = 'Z'

    #procedure standard 
    m.fit(ds_sal_temp, features=features_in_ds, dim=features_zdim)
    ds_sal_temp['TEMP'].attrs['feature_name'] = 'temperature'
    ds_sal_temp['PSAL'].attrs['feature_name'] = 'salinity'
    ds_sal_temp['DEPTH'].attrs['axis'] = 'Z'
    m.predict(ds_sal_temp, features=features_in_ds, dim=features_zdim,inplace=True)
    m.predict_proba(ds_sal_temp, features=features_in_ds, inplace=True)

    # mise en place des quantiles - > penser à faire un choix pour les quantiles dans la sidebar  
    for vname in ['TEMP', 'PSAL']:
        ds_sal_temp = ds_sal_temp.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int 
    np.int = np.int_
    return ds_pcm_sal_temp,ds_sal_temp, m

# pyxpcm profil temperature uniquement
def pyxpcm_temp(da, k):
    
    #copy da pour éviter ecrasement de da
    ds_temp = da.copy(deep=True)
    # redefinir temporairement np.int pour eviter erreur depreciation Numpy 1.20
    np.int = int
    element_depth = ds_temp.sel(DEPTH=ds_temp.DEPTH[-1]).DEPTH.values# récupère la valeur max de DEPTH
    z = np.arange(0.,element_depth,-10.)
    pcm_features = {'temperature': z} 
    m = pcm(K=k, features=pcm_features, maxvar=2)  # créer le model PCM

    features_zdim='DEPTH'
    features_in_ds = {'temperature': 'TEMP'}
    #PCM propercies
    ds_pcm_temp = ds_temp.pyxpcm.fit_predict(m, features=features_in_ds, dim=features_zdim, inplace=True)
    ds_pcm_temp['TEMP'].attrs['feature_name'] = 'temperature'
    ds_pcm_temp['DEPTH'].attrs['axis'] = 'Z'

    #procedure standard 
    m.fit(ds_temp, features=features_in_ds, dim=features_zdim)
    ds_temp['TEMP'].attrs['feature_name'] = 'temperature'
    ds_temp['DEPTH'].attrs['axis'] = 'Z'
    m.predict(ds_temp, features=features_in_ds, dim=features_zdim,inplace=True)
    m.predict_proba(ds_temp, features=features_in_ds, inplace=True)

    # mise en place des quantiles
    for vname in ['TEMP']:
        ds_temp = ds_temp.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int
    np.int = np.int_
    return ds_pcm_temp,ds_temp, m

# pyxpcm profil salinité changer da en  da_resultat
def pyxpcm_sal(da, k):

    #copy da pour éviter ecrasement de da
    ds_sal = da.copy(deep=True)
    # redefinir temporairement np.int pour eviter erreur depreciation Numpy 1.20
    np.int = int
    element_depth = ds_sal.sel(DEPTH=ds_sal.DEPTH[-1]).DEPTH.values# récupère la valeur max de DEPTH
    z = np.arange(0.,element_depth,-10.)
    pcm_features = {'salinity':z} 
    m = pcm(K=k, features=pcm_features, maxvar=2) # créer le model PCM
    
    features_zdim='DEPTH'
    features_in_ds = {'salinity': 'PSAL'}
    #PCM propercies
    ds_pcm_sal = ds_sal.pyxpcm.fit_predict(m, features=features_in_ds, dim=features_zdim, inplace=True)
    ds_pcm_sal['TEMP'].attrs['feature_name'] = 'temperature'
    ds_pcm_sal['PSAL'].attrs['feature_name'] = 'salinity'
    ds_pcm_sal['DEPTH'].attrs['axis'] = 'Z'

    #procedure standard 
    m.fit(ds_sal, features=features_in_ds, dim=features_zdim)
    ds_sal['PSAL'].attrs['feature_name'] = 'salinity'
    ds_sal['DEPTH'].attrs['axis'] = 'Z'
    m.predict(ds_sal, features=features_in_ds, dim=features_zdim,inplace=True)
    m.predict_proba(ds_sal, features=features_in_ds, inplace=True)

    # mise en place des quantiles
    for vname in ['PSAL']:
        ds_sal = ds_sal.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int 
    np.int = np.int_
    return ds_pcm_sal,ds_sal, m




# variable global lors d'une session
if 'ds' not in st.session_state:
    st.session_state.ds = xr.Dataset()

if 'ds_py' not in st.session_state:
    st.session_state.ds_py = xr.Dataset()

if 'df_points' not in st.session_state:
    st.session_state.df_points = pd.DataFrame()

if 'button_fetch_data_pressed' not in st.session_state:
    st.session_state.button_fetch_data_pressed = False


if 'button_class_data_pressed' not in st.session_state:
    st.session_state.button_class_data_pressed = False

if 'm' not in st.session_state:
    st.session_state.m = None

if 'ds_pcm' not in st.session_state:
    st.session_state.ds_pcm = xr.Dataset()

def main():
    global ds
    global df_points
    #decentre les éléments de la page 
    st.set_page_config(layout="wide")

    # parameters canvas
    # argopy parameters
    argopy_text = '<b><font color="blue" size="5">parametre argopy</font></b>'
    st.sidebar.markdown(argopy_text, unsafe_allow_html=True)

    longitude = st.sidebar.slider("longitude", min_value=-180.0, max_value=180.0, value=[-75.0, -45.0])
    latitude = st.sidebar.slider("Latitude ", min_value=-90.0, max_value=90.0, value=[30.0, 20.0])
    profondeur = st.sidebar.slider("Profondeur", min_value=0.0, max_value=6000.0, value=[0.0, 1000.0])
    interpolation = st.sidebar.slider("intervalle interpolation", min_value=1, max_value=10, value=5)
    date_debut = st.sidebar.date_input("date début", (dt.date(2010, 1, 1)), format="YYYY-MM-DD")
    date_fin = st.sidebar.date_input("date fin", (dt.date(2010, 12, 1)), format="YYYY-MM-DD")
    llon = longitude[0]
    rlon = longitude[1]
    llat = latitude[0]
    ulat = latitude[1]
    depthmin = profondeur[0]
    depthmax = profondeur[1]
    time_in = date_debut.strftime("%Y-%m-%d")
    time_f = date_fin.strftime("%Y-%m-%d")

    #button active la récupération argopy
    button_fetch_data = st.sidebar.button("Récupérer les données")

    # pyxpcm parameters
    pyxpcm_text = '<b><font color="blue" size="5">parametre pyxpcm</font></b>'
    st.sidebar.markdown(pyxpcm_text, unsafe_allow_html=True)
    clusters = st.sidebar.slider("nombre de clusters(K)", min_value=2, max_value=20, value=6)
    prof_salinite = st.sidebar.checkbox('salinite', value=True)
    prof_temperature =st.sidebar.checkbox('temperature',value= True)

    #button active la classification des données argopy
    button_class_data = st.sidebar.button("classifier les données")

    if button_fetch_data:
        st.session_state.ds  = recup_argo_data(llon,rlon, llat,ulat, depthmin, depthmax,interpolation,time_in,time_f)
        st.session_state.button_class_data_pressed = False
        st.session_state.button_fetch_data_pressed = True
        

    if button_class_data:
        if st.session_state.ds != None :
            if (prof_salinite) and (prof_temperature):
                st.session_state.ds_pcm, st.session_state.ds_py , st.session_state.m = pyxpcm_sal_temp(st.session_state.ds, clusters )
            elif (prof_salinite) and (not prof_temperature):
                st.session_state.ds_pcm, st.session_state.ds_py, st.session_state.m = pyxpcm_sal(st.session_state.ds, clusters )
            else:
                st.session_state.ds_pcm, st.session_state.ds_py, st.session_state.m = pyxpcm_temp(st.session_state.ds, clusters )

            ds_trier = st.session_state.ds_py.copy(deep =True)
            ds_trier = ds_trier.isel(DEPTH=0)
            ds_trier = ds_trier.isel(quantile=1)
            st.session_state.df_points= ds_trier.to_dataframe()
            st.session_state.button_fetch_data_pressed = False
            st.session_state.button_class_data_pressed = True
        else : 
            st.write("pas de donner argo à classifier !")


    # Map canvas
    # créer un map folium avec la latitude et longitude 
    map = folium.Map(location=[np.mean(latitude), np.mean(longitude)], zoom_start=4)

    #maj emplacement si changement latitude ou longitude
    map.location = [np.mean(latitude), np.mean(longitude)]

    # ajouter un rectangle pour marquer l'emplacement des données.
    folium.Rectangle(bounds=[(latitude[0], longitude[0]), (latitude[1], longitude[1])],
                        color='red').add_to(map)

    # Ajouter une couche de marqueurs si le bouton "Récupérer les données" a été pressé
    if st.session_state.button_fetch_data_pressed:
        
        ds_argo = st.session_state.ds.copy(deep=True)
        #st.write("nombre de profils : " + str(len(ds_argo['N_PROF'])))
        ds_argo= ds_argo.isel(DEPTH=0)
        for index, row in ds_argo.to_dataframe().iterrows():
            # Créez un popup avec les informations du DataFrame
            popup_content = """
                LATITUDE : {}<br>
                LONGITUDE : {}<br>
                TIME : {}<br>
                PSAL : {}<br>
                TEMP : {}<br>
            """.format(row['LATITUDE'], row['LONGITUDE'], row['TIME'], row['PSAL'], row['TEMP'])

            # Ajouter les marqueurs bleues
            folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],
                                radius=3,  # taille du cercle 
                                color='blue',  # couleur du cercle
                                fill=True,
                                fill_color='blue',  # couleur de remplissage du cercle
                                fill_opacity=1,
                                popup=folium.Popup(popup_content, max_width=300)).add_to(map)


    # Vérifier si le boutton classification est presser
    if st.session_state.button_class_data_pressed :
        #max_pcm_class = st.session_state.df_points['PCM_LABELS'].max()
        #st.session_state.df_points['Color'] = st.session_state.df_points['PCM_LABELS'].apply(lambda x: "#{:06x}".format((x * 977) % 0x1000000))
        for index, row in st.session_state.df_points.iterrows():
            #choisie la couleurs du PCM_LABELS aléatoirement
            
            pcm_labels_value = row['PCM_LABELS']

            if not math.isnan(pcm_labels_value):
                color = int_to_rgb(int(pcm_labels_value))
            else:
                color = 'blue'  


            # Créer un popup avec les informations du DataFrame
            popup_content = """
                    DEPTH: {}<br> 
                    PCM_CLASS: {}<br>
                    LATITUDE : {}<br>
                    LONGITUDE : {}<br>
                    TIME : {}<br>
                    PSAL : {}<br>
                    TEMP : {}<br>
                """.format(row['DEPTH'], row['PCM_LABELS'],row['LATITUDE'], row['LONGITUDE'], row['TIME'], row['PSAL'], row['TEMP'])
            folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],
                            radius=3,  # taille du cercle 
                            color=color,  # couleur du cercle
                            fill=True,
                            fill_color= color,  # couleur de remplissage du cercle
                            fill_opacity=1,
                            popup=folium.Popup(popup_content, max_width=300)).add_to(map)



        
    # Afficher la carte
    folium_static(map, width=1350, height=600)

    # graphique canvas
    if st.session_state.button_class_data_pressed :
        #st.write(st.session_state.ds_py)
        #graphique quantile profil temperature        
        if 'TEMP_Q' in st.session_state.ds_py:
            with st.expander("quantile temperature"):
                fig, ax = st.session_state.m.plot.quantile(st.session_state.ds_py['TEMP_Q'], maxcols=4, figsize=(10, 8), sharey=True)
                st.pyplot(fig)
                

        #gaphique quantile profil salinité
        if 'PSAL_Q' in st.session_state.ds_py:   
            with st.expander("quantile salinité"): 
                fig, ax = st.session_state.m.plot.quantile(st.session_state.ds_py['PSAL_Q'], maxcols=4, figsize=(10, 8), sharey=True)
                st.pyplot(fig)

        with st.expander("scaler propertie"):
            fig, ax = st.session_state.m.plot.scaler()
            st.pyplot(fig) 
        with st.expander("reducer properties"):
            fig, ax = st.session_state.m.plot.reducer()
            st.pyplot(fig)
        
        if ('TEMP_Q' in st.session_state.ds_py) and ('PSAL_Q' in st.session_state.ds_py):
            with st.expander("preprocesse data  1"):
                g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'temperature': 'TEMP', 'salinity': 'PSAL'}, style='darkgrid')   
                st.pyplot(g)
            with st.expander("preprocesse data 2"):
                g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'temperature': 'TEMP', 'salinity': 'PSAL'},kde=True)
                st.pyplot(g)
        elif 'TEMP_Q' not in st.session_state.ds_py :
            with st.expander("preprocesse data  1"):
                g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'salinity': 'PSAL'}, style='darkgrid')   
                st.pyplot(g)
            with st.expander("preprocesse data 2"):
                g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'salinity': 'PSAL'},kde=True)
                st.pyplot(g)
        else :
            with st.expander("preprocesse data  1"):
                g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'temperature': 'TEMP'}, style='darkgrid')   
                st.pyplot(g)
            with st.expander("preprocesse data 2"):
                g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'temperature': 'TEMP'},kde=True)
                st.pyplot(g)

        colum_x = st.selectbox('Sélectionner x', st.session_state.ds_py.to_dataframe().columns)
        ds_data = st.session_state.ds_py.copy(deep =True)
        ds_data = ds_data.isel(DEPTH=0)
        st.write(ds_data)
        # Récupérer les données de la colonne sélectionnée
        data_x = ds_data[colum_x].values

        # Afficher les données de la colonne sélectionnée
        st.write(f"Vous avez sélectionné la colonne : {colum_x}")
        st.write(data_x)

        # Tracer un graphique avec Matplotlib
        fig, ax = plt.subplots()
        ax.plot(data_x)
        ax.plot(range(0,1000, 5), [mean(st.session_state.ds_py.isel(DEPTH=x))[colum_x].values for x in range(0,1000, 5)])
        ax.set_xlabel('Index')
        ax.set_ylabel('Valeurs')
        ax.set_title(f'Données de {colum_x}')
        st.pyplot(fig)
   








if __name__ == "__main__":
    main()
