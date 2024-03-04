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
import plotly.express as px
import plotly.graph_objects as go

def afficher_graphiques_interactifs(df_points):
    if df_points.empty:
        return
    # Créer un scatter plot interactif avec les clusters colorés
    fig = px.scatter(df_points, x='TEMP', y='PSAL', color='PCM_LABELS',
                     hover_data=['PSAL', 'LATITUDE', 'LONGITUDE', 'TIME'],
                     title="Graphique interactif après clustering",
                     labels={'TEMP': 'Temperature', 'PSAL': 'salinité'},
                     width=800, height=600)

    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

def get_pcm_labels_info(pcm_labels_value,pcm_labels_info):

    if pcm_labels_value in pcm_labels_info:
        return pcm_labels_info[pcm_labels_value]
    else:
        # Valeurs par défaut si la classe n'est pas trouvée
        return {
            'moyenne_salinite': None,
            'temperature': None,
            'latitude': None,
            'longitude': None
        }



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
def pyxpcm_sal_temp(da, k,quan, varmax ):

    #copy da pour éviter ecrasement de da
    ds_sal_temp = da.copy(deep=True)
    # redefinir temporairement np.int pour eviter erreur depreciation Numpy 1.20
    np.int = int
    element_depth = ds_sal_temp.sel(DEPTH=ds_sal_temp.DEPTH[-1]).DEPTH.values# récupère la valeur max de DEPTH
    z = np.arange(0.,element_depth,-10.) 
    pcm_features = {'temperature': z, 'salinity':z} 
    m = pcm(K=k, features=pcm_features, maxvar=varmax) # créer le model PCM
    
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


    # Utiliser la liste de quantiles dans la fonction
    for vname in ['TEMP', 'PSAL']:
        ds_sal_temp = ds_sal_temp.pyxpcm.quantile(m, q=quan, of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int 
    np.int = np.int_
    ds_sal_temp['PROFIL'] = 'PSAL & TEMP'
    return ds_pcm_sal_temp,ds_sal_temp, m

# pyxpcm profil temperature uniquement
def pyxpcm_temp(da, k, quan, varmax):
    
    #copy da pour éviter ecrasement de da
    ds_temp = da.copy(deep=True)
    # redefinir temporairement np.int pour eviter erreur depreciation Numpy 1.20
    np.int = int
    element_depth = ds_temp.sel(DEPTH=ds_temp.DEPTH[-1]).DEPTH.values# récupère la valeur max de DEPTH
    z = np.arange(0.,element_depth,-10.)
    pcm_features = {'temperature': z} 
    m = pcm(K=k, features=pcm_features, maxvar=varmax)  # créer le model PCM

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
        ds_temp = ds_temp.pyxpcm.quantile(m, q=quan, of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int
    np.int = np.int_
    ds_temp['PROFIL'] = 'TEMP'
    return ds_pcm_temp,ds_temp, m

# pyxpcm profil salinité changer da en  da_resultat
def pyxpcm_sal(da, k, quan, varmax):

    #copy da pour éviter ecrasement de da
    ds_sal = da.copy(deep=True)
    # redefinir temporairement np.int pour eviter erreur depreciation Numpy 1.20
    np.int = int
    element_depth = ds_sal.sel(DEPTH=ds_sal.DEPTH[-1]).DEPTH.values# récupère la valeur max de DEPTH
    z = np.arange(0.,element_depth,-10.)
    pcm_features = {'salinity':z} 
    m = pcm(K=k, features=pcm_features, maxvar=varmax) # créer le model PCM
    
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
        ds_sal = ds_sal.pyxpcm.quantile(m, q=quan, of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int 
    np.int = np.int_
    ds_sal['PROFIL'] = 'PSAL'
    return ds_pcm_sal,ds_sal, m




# variable global lors d'une session
if 'ds' not in st.session_state:
    st.session_state.ds = xr.Dataset()

if 'ds_py' not in st.session_state:
    st.session_state.ds_py = xr.Dataset()

if 'df_points' not in st.session_state:
    st.session_state.df_points = pd.DataFrame()

if 'pcm_labels_info' not in st.session_state:
    st.session_state.pcm_labels_info = None

if 'button_fetch_data_pressed' not in st.session_state:
    st.session_state.button_fetch_data_pressed = False


if 'button_class_data_pressed' not in st.session_state:
    st.session_state.button_class_data_pressed = False

if 'm' not in st.session_state:
    st.session_state.m = None

if 'ds_pcm' not in st.session_state:
    st.session_state.ds_pcm = xr.Dataset()

if 'graphs_updated' not in st.session_state:
    st.session_state.graphs_updated = False

if 'graphs' not in st.session_state:
    st.session_state.graphs = {
        'quantile_temp': None,
        'quantile_sal': None,
        'scaler': None,
        'reducer': None,
        'preprocess': None
    }

def main():
    global ds
    global df_points
    
    #decentre les éléments de la page 
    st.set_page_config(layout="wide", page_title= "argoPXPCM", page_icon=":anchor:")

    # parameters canvas
    # argopy parameters
    argopy_text = '<b><font color="blue" size="5">parametre argopy</font></b>'
    st.sidebar.markdown(argopy_text, unsafe_allow_html=True)
    with st.sidebar.expander("zone de sélection"):
        longitude = st.slider("longitude", min_value=-180.0, max_value=180.0, value=[-75.0, -45.0])
        latitude = st.slider("Latitude ", min_value=-90.0, max_value=90.0, value=[30.0, 20.0])
        profondeur = st.slider("Profondeur", min_value=0.0, max_value=6000.0, value=[0.0, 1000.0])
    with  st.sidebar.expander("date de sélection"):
        date_debut = st.date_input("date début", (dt.date(2010, 1, 1)), format="YYYY-MM-DD")
        date_fin = st.date_input("date fin", (dt.date(2010, 12, 1)), format="YYYY-MM-DD")
    with st.sidebar.expander("interpolation de selection"):
        interpolation = st.slider("intervalle interpolation", min_value=1, max_value=10, value=5)
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
    with st.sidebar.expander("profil clustering"):
        clusters = st.slider("nombre de clusters(K)", min_value=2, max_value=20, value=6)
        prof_salinite = st.checkbox('salinite', value=True)
        prof_temperature =st.checkbox('temperature',value= True)
        maxVARS = st.number_input("maxvar", value=2, min_value=2, max_value=20)
    with st.sidebar.expander("quantile"):
        # Nombre initial quantile
        num_entries = st.number_input("Nombre quantiles", min_value=1, max_value=10, step=1, value=3)
        # valeurs par défaut pour les quantiles
        quantiles = []
        default_values = [0.05, 0.5, 0.95]  
        for i in range(num_entries):
            default_value = default_values[i] if i < len(default_values) else 0.00  # si on dépasse les valeurs par défaut, revenir à 0.05
            quantile_input = st.number_input(f"quantile {i+1}", value=default_value)
            quantiles.append(quantile_input)
    #button active la classification des données argopy
    button_class_data = st.sidebar.button("classifier les données")

    #graphique paramètres
    graphique_text = '<b><font color="blue" size="5">paramètres graphique</font></b>'
    st.sidebar.markdown(graphique_text, unsafe_allow_html=True)
    with st.sidebar.expander("quantile graphique"):
        graph_quan_ok = st.checkbox("afficher graphique quantile", value=True)
        if graph_quan_ok:
            quan_maxcols = st.number_input("maximum colonne", value= 4 )
            quan_figsizex = st.number_input("taille x des case", value= 10)
            quan_figsizey = st.number_input("taille y des cases", value=8)

    with st.sidebar.expander("scaler graphique(means and std)"):
        graph_scal_ok =  st.checkbox("afficher graphique scaler",value= True)
        if graph_scal_ok:
            scal_style = st.selectbox("style scaler",('whitegrid','darkgrid','white','dark'))
            scal_colorLine = st.selectbox("color line scaler",('red','blue','green','cyan', 'magenta', 'yellow', 'black'))
            scal_sizeLine = st.number_input("size line scaler", value=1.5, max_value= 4.0, min_value=0.5)
            scal_axeY = st.slider("marge Y scaler", min_value=-depthmin, max_value=-depthmax, value=[-depthmax, -depthmin])

    with st.sidebar.expander("reducer"):
        graph_redu_ok =  st.checkbox("afficher graphique reducer",value= True)
        if graph_redu_ok:
            redu_PCA = st.number_input("number of PCA ", min_value=1, max_value=maxVARS, value=2)
            redu_style = st.selectbox("style reducer",('whitegrid','darkgrid','white','dark'))
            redu_sizeLine = st.number_input("size line reducer", value=1.5, max_value= 4.0, min_value=0.5)
            redu_axeY = st.slider("marge Y reducer", min_value=-depthmin, max_value=-depthmax, value=[-depthmax, -depthmin])

    
    with st.sidebar.expander("pre-process data "):
        graph_pro_ok = st.checkbox("afficher graphique pre-process data",value= False)
        if graph_pro_ok :
            pro_hist_ok = st.checkbox("afficher avec histgramme ? attention lent", value=False)

    if st.sidebar.button("Afficher les graphiques"):
        st.session_state.graphs_updated = True 



    if button_fetch_data:
        st.session_state.ds  = recup_argo_data(llon,rlon, llat,ulat, depthmin, depthmax,interpolation,time_in,time_f)
        st.session_state.button_class_data_pressed = False
        st.session_state.button_fetch_data_pressed = True
        

    if button_class_data:
        if st.session_state.ds != None :
            if (prof_salinite) and (prof_temperature):
                st.session_state.ds_pcm, st.session_state.ds_py , st.session_state.m = pyxpcm_sal_temp(st.session_state.ds, clusters, quantiles, maxVARS )
            elif (prof_salinite) and (not prof_temperature):
                st.session_state.ds_pcm, st.session_state.ds_py, st.session_state.m = pyxpcm_sal(st.session_state.ds, clusters, quantiles, maxVARS )
            else:
                st.session_state.ds_pcm, st.session_state.ds_py, st.session_state.m = pyxpcm_temp(st.session_state.ds, clusters, quantiles, maxVARS )

            ds_trier = st.session_state.ds_py.copy(deep =True)
            ds_trier = ds_trier.isel(DEPTH=0)
            ds_trier = ds_trier.isel(quantile=1)
            st.session_state.df_points= ds_trier.to_dataframe()
            #création dictionnaire info class 
            # Calcul des moyennes de salinité et de température par classe PCM_LABELS
            pcm_labels_groups =st.session_state.df_points.copy(deep =True)
            pcm_labels_groups = pcm_labels_groups.groupby('PCM_LABELS')
            pcm_labels_moyennes = pcm_labels_groups.agg({
                'PSAL': 'mean',
                'TEMP': 'mean',
                'LATITUDE': 'mean',
                'LONGITUDE': 'mean'
            }).reset_index()

            st.session_state.pcm_labels_info = {}
            for index, row in pcm_labels_moyennes.iterrows():
                pcm_label = row['PCM_LABELS']
                st.session_state.pcm_labels_info[pcm_label] = {
                    'moyenne_salinite': row['PSAL'],
                    'moyenne_temperature': row['TEMP'],
                    'latitude': row['LATITUDE'],
                    'longitude': row['LONGITUDE']
                }
            
            
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

            pcm_info = get_pcm_labels_info(pcm_labels_value,st.session_state.pcm_labels_info)

            # Créer un popup avec les informations du DataFrame
            popup_content = """
                    DEPTH: {}<br> 
                    LATITUDE : {}<br>
                    LONGITUDE : {}<br>
                    TIME : {}<br>
                    PSAL : {}<br>
                    TEMP : {}<br>
                    PROFIL : {}<br>
                    <br>
                    PCM_CLASS: {}<br>
                    PSAL MOY CLASS: {:.2f}<br>
                    TEMP MOY CLASS: {:.2f}<br>
                    LATITUDE MOY CLASS: {:.2f}<br>
                    LONGITUDE MOY CLASS: {:.2f}<br>
                """.format(row['DEPTH'],row['LATITUDE'], row['LONGITUDE'], row['TIME'], row['PSAL'], row['TEMP'], row['PROFIL'], row['PCM_LABELS'],
               pcm_info['moyenne_salinite'], pcm_info['moyenne_temperature'],
               pcm_info['latitude'], pcm_info['longitude'])
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
        if st.session_state.graphs_updated:
            
            #st.write(st.session_state.ds_py)
            #graphique quantile profil temperature        
            if ('TEMP_Q' in st.session_state.ds_py) and (graph_quan_ok):
                with st.expander("quantile temperature"):
                    fig, ax = st.session_state.m.plot.quantile(st.session_state.ds_py['TEMP_Q'], maxcols=quan_maxcols, figsize=(quan_figsizex, quan_figsizey), sharey=True)
                    st.session_state.graphs['quantile_temp'] = fig
            elif not graph_quan_ok:
               st.session_state.graphs['quantile_temp'] = None
                    
            #graphique quantile profil salinité
            if ('PSAL_Q' in st.session_state.ds_py) and (graph_quan_ok):   
                with st.expander("quantile salinité"): 
                    fig, ax = st.session_state.m.plot.quantile(st.session_state.ds_py['PSAL_Q'],maxcols=quan_maxcols, figsize=(quan_figsizex, quan_figsizey), sharey=True)
                    st.session_state.graphs['quantile_sal'] = fig
            elif not graph_quan_ok:
               st.session_state.graphs['quantile_temp'] = None

            if graph_scal_ok:
                with st.expander("scaler propertie"):
                    fig, ax = st.session_state.m.plot.scaler(style =scal_style,plot_kw={'color': scal_colorLine, 'linewidth': scal_sizeLine},subplot_kw={'ylim':[scal_axeY[0],scal_axeY[1]]} )
                    st.session_state.graphs['scaler'] = fig
            else:
                st.session_state.graphs['scaler'] = None

            if graph_redu_ok : 
                with st.expander("reducer properties"):
                    fig, ax = st.session_state.m.plot.reducer(pcalist = range(0,redu_PCA),style =redu_style,plot_kw={'linewidth': redu_sizeLine},subplot_kw={'ylim':[redu_axeY[0],redu_axeY[1]]} )
                    st.session_state.graphs['reducer'] = fig
            else: 
                st.session_state.graphs['reducer'] = None

            if graph_pro_ok :
                if ('TEMP_Q' in st.session_state.ds_py) and ('PSAL_Q' in st.session_state.ds_py):
                    if not pro_hist_ok :
                        with st.expander("pre_process data"):
                            g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'temperature': 'TEMP', 'salinity': 'PSAL'}, style='darkgrid')   
                            st.session_state.graphs['preprocess'] = g

                    if pro_hist_ok :
                        with st.expander("pre_processe data"):
                            g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'temperature': 'TEMP', 'salinity': 'PSAL'},kde=True)
                            st.session_state.graphs['preprocess'] = g

                elif 'TEMP_Q' not in st.session_state.ds_py :
                    if not pro_hist_ok :
                        with st.expander("pre-process data"):
                            g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'salinity': 'PSAL'}, style='darkgrid')   
                            st.session_state.graphs['preprocess'] = g

                    if pro_hist_ok :
                        with st.expander("pre-process data"):
                            g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'salinity': 'PSAL'},kde=True)
                            st.session_state.graphs['preprocess'] = g

                else :
                    if not pro_hist_ok:
                        with st.expander("pre-process data"):
                            g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'temperature': 'TEMP'}, style='darkgrid')   
                            st.session_state.graphs['preprocess'] = g

                    if pro_hist_ok:
                        with st.expander("pre-process data"):
                            g = st.session_state.m.plot.preprocessed(st.session_state.ds_pcm, features={'temperature': 'TEMP'},kde=True)
                            st.session_state.graphs['preprocess'] = g
            else :
                st.session_state.graphs['preprocess'] = None

        st.session_state.graphs_updated = False

        # Afficher les graphiques à partir de st.session_state.graphs
    if st.session_state.graphs['quantile_temp'] is not None:
        with st.expander("quantile temperature"):
            st.pyplot(st.session_state.graphs['quantile_temp'])
                
    if st.session_state.graphs['quantile_sal'] is not None:
        with st.expander("quantile salinité"):
            st.pyplot(st.session_state.graphs['quantile_sal'])
            
    if st.session_state.graphs['scaler'] is not None:
        with st.expander("scaler propertie"):
            st.pyplot(st.session_state.graphs['scaler'])
            
    if st.session_state.graphs['reducer'] is not None:
        with st.expander("reducer properties"):
            st.pyplot(st.session_state.graphs['reducer'])
                
    if st.session_state.graphs['preprocess'] is not None:
        with st.expander("pre-process data"):
            st.pyplot(st.session_state.graphs['preprocess'])
    
    ds_test = st.session_state.ds_py.copy(deep =True)
    ds_test = ds_test.isel(DEPTH=0)
    ds_test = ds_test.isel(quantile=0)
    ds_test= ds_test.to_dataframe()
    afficher_graphiques_interactifs(ds_test)

        









if __name__ == "__main__":
    main()
