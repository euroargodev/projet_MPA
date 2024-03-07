# Main file to make the application work
# to start streamlit: streamlit run main.py
# to activate MPA environment: conda activate MPA
# to generate environment.yml: conda env export > environment.yml
# to set up MPA environment on a new computer with the environment.yml file: conda env create -f environment.yml
# to open anaconda navigator (environment manager): anaconda-navigator

from statistics import mean
import gsw
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 
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


#def scaler_interactif(ds):
    

def afficher_graphiques_interactifs(df_points):
    if df_points.empty:
        return
    # Create an interactive scatter plot with colored clusters
    fig = px.scatter(df_points, x='TEMP', y='PSAL', color='PCM_LABELS',
                     hover_data=['PSAL', 'LATITUDE', 'LONGITUDE', 'TIME'],
                     title="Graphique interactif après clustering",
                     labels={'TEMP': 'Temperature', 'PSAL': 'salinité'},
                     width=800, height=600)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

#Display pcm_info
def get_pcm_labels_info(pcm_labels_value,pcm_labels_info):

    if pcm_labels_value in pcm_labels_info:
        return pcm_labels_info[pcm_labels_value]
    else:
        # Default values
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

    # Temporarily redefine np.int to int to avoid deprecated numpy warning
    np.int = int
    # Add a safety margin to avoid excluding profiles
    depthmaxsecu= depthmax + 100
    # Retrieve data using argopy
    ds_points = ArgoDataFetcher(src='erddap', parallel=True).region([llon,rlon, llat,ulat, depthmin, depthmaxsecu,time_in,time_f]).to_xarray()
    
    # Convert to 2 dimensions (N_PROF x N_LEVELS)
    ds = ds_points.argo.point2profile()
    # Retrieve SIG0 and N2(BRV2) data using TEOS-10
    ds.argo.teos10(['SIG0','N2'])

    # z is created to represent depths from minimum to maximum depth with an interval 
    z=np.arange(depthmin,depthmax,intervalle)
     # Interpolate with depths z
    ds2 = ds.argo.interp_std_levels(z)

    # Calculate depth with gsw.z_from_p from p (PRES) and lat (LATITUDE)
    p=np.array(ds2.PRES)
    lat=np.array(ds2.LATITUDE)
    z=np.ones_like(p)
    nprof=np.array(ds2.N_PROF)

    for i in np.arange(0,len(nprof)):
        z[i,:]=gsw.z_from_p(p[i,:], lat[i])


    # Calculate depth from interpolated pressure
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


# pyxpcm profile salinity and temperature
def pyxpcm_sal_temp(da, k, quan, varmax):

    # copy da to avoid overwriting
    ds_sal_temp = da.copy(deep=True)
    # redefine np.int temporarily to avoid Numpy 1.20 deprecation warning
    np.int = int
    element_depth = ds_sal_temp.sel(DEPTH=ds_sal_temp.DEPTH[-1]).DEPTH.values  # retrieves the max value of DEPTH
    z = np.arange(0., element_depth, -10.) 
    pcm_features = {'temperature': z, 'salinity': z} 
    m = pcm(K=k, features=pcm_features, maxvar=varmax)  # create the PCM model
    
    features_zdim = 'DEPTH'
    features_in_ds = {'temperature': 'TEMP', 'salinity': 'PSAL'}
    # PCM properties
    ds_pcm_sal_temp = ds_sal_temp.pyxpcm.fit_predict(m, features=features_in_ds, dim=features_zdim, inplace=True)
    ds_pcm_sal_temp['TEMP'].attrs['feature_name'] = 'temperature'
    ds_pcm_sal_temp['PSAL'].attrs['feature_name'] = 'salinity'
    ds_pcm_sal_temp['DEPTH'].attrs['axis'] = 'Z'

    # standard procedure 
    m.fit(ds_sal_temp, features=features_in_ds, dim=features_zdim)
    ds_sal_temp['TEMP'].attrs['feature_name'] = 'temperature'
    ds_sal_temp['PSAL'].attrs['feature_name'] = 'salinity'
    ds_sal_temp['DEPTH'].attrs['axis'] = 'Z'
    m.predict(ds_sal_temp, features=features_in_ds, dim=features_zdim, inplace=True)
    m.predict_proba(ds_sal_temp, features=features_in_ds, inplace=True)

    # Use the quantile list in the function
    for vname in ['TEMP', 'PSAL']:
        ds_sal_temp = ds_sal_temp.pyxpcm.quantile(m, q=quan, of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int 
    np.int = np.int_
    ds_sal_temp['PROFIL'] = 'PSAL & TEMP'
    return ds_pcm_sal_temp, ds_sal_temp, m

# pyxpcm profile temperature only
def pyxpcm_temp(da, k, quan, varmax):
    
    # copy da to avoid overwriting
    ds_temp = da.copy(deep=True)
    # redefine np.int temporarily to avoid Numpy 1.20 deprecation warning
    np.int = int
    element_depth = ds_temp.sel(DEPTH=ds_temp.DEPTH[-1]).DEPTH.values  # retrieves the max value of DEPTH
    z = np.arange(0., element_depth, -10.)
    pcm_features = {'temperature': z} 
    m = pcm(K=k, features=pcm_features, maxvar=varmax)  # create the PCM model

    features_zdim = 'DEPTH'
    features_in_ds = {'temperature': 'TEMP'}
    # PCM properties
    ds_pcm_temp = ds_temp.pyxpcm.fit_predict(m, features=features_in_ds, dim=features_zdim, inplace=True)
    ds_pcm_temp['TEMP'].attrs['feature_name'] = 'temperature'
    ds_pcm_temp['DEPTH'].attrs['axis'] = 'Z'

    # standard procedure 
    m.fit(ds_temp, features=features_in_ds, dim=features_zdim)
    ds_temp['TEMP'].attrs['feature_name'] = 'temperature'
    ds_temp['DEPTH'].attrs['axis'] = 'Z'
    m.predict(ds_temp, features=features_in_ds, dim=features_zdim, inplace=True)
    m.predict_proba(ds_temp, features=features_in_ds, inplace=True)

    # setting up quantiles
    for vname in ['TEMP']:
        ds_temp = ds_temp.pyxpcm.quantile(m, q=quan, of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int
    np.int = np.int_
    ds_temp['PROFIL'] = 'TEMP'
    return ds_pcm_temp, ds_temp, m

# pyxpcm profile salinity change da to da_result
def pyxpcm_sal(da, k, quan, varmax):

    # copy da to avoid overwriting
    ds_sal = da.copy(deep=True)
    # redefine np.int temporarily to avoid Numpy 1.20 deprecation warning
    np.int = int
    element_depth = ds_sal.sel(DEPTH=ds_sal.DEPTH[-1]).DEPTH.values  # retrieves the max value of DEPTH
    z = np.arange(0., element_depth, -10.)
    pcm_features = {'salinity': z} 
    m = pcm(K=k, features=pcm_features, maxvar=varmax)  # create the PCM model
    
    features_zdim = 'DEPTH'
    features_in_ds = {'salinity': 'PSAL'}
    # PCM properties
    ds_pcm_sal = ds_sal.pyxpcm.fit_predict(m, features=features_in_ds, dim=features_zdim, inplace=True)
    ds_pcm_sal['TEMP'].attrs['feature_name'] = 'temperature'
    ds_pcm_sal['PSAL'].attrs['feature_name'] = 'salinity'
    ds_pcm_sal['DEPTH'].attrs['axis'] = 'Z'

    # standard procedure 
    m.fit(ds_sal, features=features_in_ds, dim=features_zdim)
    ds_sal['PSAL'].attrs['feature_name'] = 'salinity'
    ds_sal['DEPTH'].attrs['axis'] = 'Z'
    m.predict(ds_sal, features=features_in_ds, dim=features_zdim, inplace=True)
    m.predict_proba(ds_sal, features=features_in_ds, inplace=True)

    # setting up quantiles
    for vname in ['PSAL']:
        ds_sal = ds_sal.pyxpcm.quantile(m, q=quan, of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    # Reset np.int 
    np.int = np.int_
    ds_sal['PROFIL'] = 'PSAL'
    return ds_pcm_sal, ds_sal, m


# Global variables during a session
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
    
     # Decenter the elements of the page
    st.set_page_config(layout="wide", page_title= "argoPXPCM", page_icon=":anchor:")

    # Sidebar
    # Argopy parameters
    argopy_text = '<b><font color="blue" size="5">Argopy parameters</font></b>'
    st.sidebar.markdown(argopy_text, unsafe_allow_html=True)
    with st.sidebar.expander("Selection zone"):
        latitude_option = st.selectbox("Latitude choice", ("Slider", "manual"))
        if latitude_option == "Slider":
            latitude = st.slider("Latitude", min_value=-90.0, max_value=90.0, value=[30.0, 20.0])
            llat = latitude[0]
            ulat = latitude[1]
        else:
            latitude_max = st.number_input("Max latitude", min_value=-90.0, max_value=90.0, value=30.0)
            latitude_min = st.number_input("Min latitude", min_value=-90.0, max_value=90.0, value=20.0)
            if ( latitude_max < latitude_min):
                st.write("Latitude max must be greater than Latitude min")
            else :
                latitude = [latitude_max,latitude_min]
                llat = latitude_min
                ulat = latitude_max
        longitude_option = st.selectbox("Longitude Choice", ("Slider", "manual"))
        if longitude_option == "Slider":
            longitude = st.slider("Longitude", min_value=-180.0, max_value=180.0, value=[-75.0, -45.0])
            llon = longitude[0]
            rlon = longitude[1]
        else:
            longitude_max = st.number_input("Max longitude", min_value=-180.0, max_value=180.0, value=-45.0)
            longitude_min = st.number_input("Min longitude", min_value=-180.0, max_value=180.0, value=-75.0)
            if ( longitude_max < longitude_min):
                st.write("Longitude max must be greater than Longitude min")
            else :
                longitude = [longitude_max,longitude_min]
                llon = longitude_min
                rlon = longitude_max
        depth_option = st.selectbox("Depth Choice", ("Slider", "manual"))
        if depth_option == "Slider":
            profondeur = st.slider("Depth", min_value=0.0, max_value=6000.0, value=[0.0, 1000.0])
            depthmin = profondeur[0]
            depthmax = profondeur[1]
        else:
            profondeur_max = st.number_input("Max depth ", min_value=0.0, max_value=6000.0, value=1000.0)
            profondeur_min = st.number_input("Min depth", min_value=0.0, max_value=6000.0, value=0.0)
            if ( profondeur_max < profondeur_min):
                st.write("Depth max must be greater than Depth min")
            else :
                    profondeur =[profondeur_min,profondeur_max]
                    depthmin = profondeur_min
                    depthmax = profondeur_max

    with  st.sidebar.expander("Date selection"):
        date_debut = st.date_input("Start date", (dt.date(2010, 1, 1)), format="YYYY-MM-DD")
        date_fin = st.date_input("End date", (dt.date(2010, 12, 31)), format="YYYY-MM-DD")
    with st.sidebar.expander("Interpolation selection"):
        interpolation = st.slider("Interpolation interval", min_value=1, max_value=10, value=5)


    time_in = date_debut.strftime("%Y-%m-%d")
    time_f = date_fin.strftime("%Y-%m-%d")

     # Button to trigger argopy data retrieval
    button_fetch_data = st.sidebar.button("Fetch data")

     # PyXPCM parameters
    pyxpcm_text = '<b><font color="blue" size="5">PyXPCM parameters</font></b>'
    st.sidebar.markdown(pyxpcm_text, unsafe_allow_html=True)
    with st.sidebar.expander("Cluster Profile"):
        clusters = st.slider("Number of clusters (K)", min_value=2, max_value=20, value=6)
        prof_salinite = st.checkbox('Salinity', value=True)
        prof_temperature =st.checkbox('Temperature',value= True)
        maxVARS = st.number_input("maxvar", value=2, min_value=2, max_value=20)
    with st.sidebar.expander("quantile"):
        # Initial number of quantiles
        num_entries = st.number_input("Number of quantiles", min_value=1, max_value=10, step=1, value=3)
        # Default values for quantiles
        quantiles = []
        default_values = [0.05, 0.5, 0.95]  
        for i in range(num_entries):
            default_value = default_values[i] if i < len(default_values) else 0.00  
            quantile_input = st.number_input(f"quantile {i+1}", value=default_value)
            quantiles.append(quantile_input)
    # Button to trigger argopy data classification
    button_class_data = st.sidebar.button("Classify Data")

    # Graphs parameters
    graphique_text = '<b><font color="blue" size="5">Graphs parameters</font></b>'
    st.sidebar.markdown(graphique_text, unsafe_allow_html=True)
    with st.sidebar.expander("Quantile Graph"):
        graph_quan_ok = st.checkbox("Show Quantile Graph", value=True)
        if graph_quan_ok:
            quan_maxcols = st.number_input("Maximum Columns", value= 4 )
            quan_figsizex = st.number_input("Figure Size X", value= 10)
            quan_figsizey = st.number_input("Figure Size Y", value=8)

    with st.sidebar.expander("Scaler Graph (Means and Std)"):
        graph_scal_ok =  st.checkbox("Show Scaler Graph",value= True)
        if graph_scal_ok:
            scal_style = st.selectbox("style scaler",('whitegrid','darkgrid','white','dark'))
            scal_colorLine = st.selectbox("color line scaler",('red','blue','green','cyan', 'magenta', 'yellow', 'black'))
            scal_sizeLine = st.number_input("size line scaler", value=1.5, max_value= 4.0, min_value=0.5)
            scal_axeY = st.slider("marge Y scaler", min_value=-depthmin, max_value=-depthmax, value=[-depthmax, -depthmin])

    with st.sidebar.expander("Reducer Graph"):
        graph_redu_ok =  st.checkbox("Show Reducer Graph ",value= True)
        if graph_redu_ok:
            redu_PCA = st.number_input("number of PCA ", min_value=1, max_value=maxVARS, value=2)
            redu_style = st.selectbox("style reducer",('whitegrid','darkgrid','white','dark'))
            redu_sizeLine = st.number_input("size line reducer", value=1.5, max_value= 4.0, min_value=0.5)
            redu_axeY = st.slider("marge Y reducer", min_value=-depthmin, max_value=-depthmax, value=[-depthmax, -depthmin])

    
    with st.sidebar.expander("Pre-process data Graph"):
        graph_pro_ok = st.checkbox("Show Pre-process data Graph",value= False)
        if graph_pro_ok :
            pro_hist_ok = st.checkbox("Show with histogramme ? Warn: slow", value=False)

    if st.sidebar.button("Show Graphs"):
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
            # Create dictionary with Class information
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
    # Create folium map 
    map = folium.Map(location=[np.mean(latitude), np.mean(longitude)], zoom_start=4)

    #change emplacement if latitude and/or longitude change
    map.location = [np.mean(latitude), np.mean(longitude)]

    # Add rectangle in red 
    folium.Rectangle(bounds=[(latitude[0], longitude[0]), (latitude[1], longitude[1])],
                        color='red').add_to(map)

    # Show Argopy data ( not classifie )
    if st.session_state.button_fetch_data_pressed:
        
        ds_argo = st.session_state.ds.copy(deep=True)
        ds_argo= ds_argo.isel(DEPTH=0)
        for index, row in ds_argo.to_dataframe().iterrows():
            # Create Popup with information about the point
            popup_content = """
                LATITUDE : {}<br>
                LONGITUDE : {}<br>
                TIME : {}<br>
                PSAL : {}<br>
                TEMP : {}<br>
            """.format(row['LATITUDE'], row['LONGITUDE'], row['TIME'], row['PSAL'], row['TEMP'])

            # Ajouter les marqueurs bleues
            folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],
                                radius=3,  
                                color='blue',  
                                fill=True,
                                fill_color='blue',  
                                fill_opacity=1,
                                popup=folium.Popup(popup_content, max_width=300)).add_to(map)


    # Show classifie data on map with information 
    if st.session_state.button_class_data_pressed :

        for index, row in st.session_state.df_points.iterrows():
            #chose PCM color random 
            
            pcm_labels_value = row['PCM_LABELS']

            if not math.isnan(pcm_labels_value):
                color = int_to_rgb(int(pcm_labels_value))
            else:
                color = 'blue'  

            pcm_info = get_pcm_labels_info(pcm_labels_value,st.session_state.pcm_labels_info)

            # Create Popup with infomartion about the point and his class
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
                            radius=3,  
                            color=color,  
                            fill=True,
                            fill_color= color,  
                            fill_opacity=1,
                            popup=folium.Popup(popup_content, max_width=300)).add_to(map)
    # Show map
    folium_static(map, width=1350, height=600)

 
    # Graphs 
   
    if st.session_state.button_class_data_pressed :
        if st.session_state.graphs_updated:
            
                   
            if ('TEMP_Q' in st.session_state.ds_py) and (graph_quan_ok):
                with st.expander("quantile temperature"):
                    fig, ax = st.session_state.m.plot.quantile(st.session_state.ds_py['TEMP_Q'], maxcols=quan_maxcols, figsize=(quan_figsizex, quan_figsizey), sharey=True)
                    st.session_state.graphs['quantile_temp'] = fig
            elif not graph_quan_ok:
               st.session_state.graphs['quantile_temp'] = None
                    
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

    # Show Graph with st.session_state.graphs
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
    

    # scaler 
    ds_data = st.session_state.ds_py.copy(deep =True)
    depth_values = ds_data.coords['DEPTH'].values
    ds_data = ds_data.to_dataframe()
    mean_psal = ds_data.groupby('DEPTH')['PSAL'].mean()
    std_psal = ds_data.groupby('DEPTH')['PSAL'].std()
    mean_temp = ds_data.groupby('DEPTH')['TEMP'].mean()
    std_temp = ds_data.groupby('DEPTH')['TEMP'].std()
    
    data = pd.DataFrame({
    'DEPTH': depth_values,
    'MEAN_PSAL': [mean_psal[depth] for depth in depth_values],
    'STD_PSAL' : [std_psal[depth] for depth in depth_values],
    'MEAN_TEMP': [mean_temp[depth] for depth in depth_values],
    'STD_TEMP' : [std_temp[depth] for depth in depth_values]

    })
    st.write(data)

    fig = px.line(data, x='MEAN_PSAL', y='DEPTH',title= "Scaler PSAL MEAN")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.line(data, x='STD_PSAL', y='DEPTH',title= "Scaler PSAL STD")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(data, x='MEAN_TEMP', y='DEPTH',title= "Scaler TEMP MEAN")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(data, x='STD_TEMP', y='DEPTH',title= "Scaler TEMP STD")
    st.plotly_chart(fig, use_container_width=True)

    
    
    #reducer PSAL
    ds_reducer = st.session_state.ds_py.copy(deep=True)
    psal_reducer = ds_reducer['PSAL'].values
    depth_reducer = ds_reducer.coords['DEPTH'].values
    pca = PCA(n_components=3, svd_solver = 'full')
    df = pca.fit(psal_reducer)
    components = pca.components_
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(components[i],depth_reducer, label=f'PCA {i+1}')
    
    ax.set_xlabel('Valeurs PCA')
    ax.set_ylabel('Profondeur')
    ax.set_title('PCA PSAL')
    ax.legend()
    st.pyplot(fig)


    #quantile 
    ds_quantile = st.session_state.ds_py.copy(deep=True)
    depth_quantile = ds_quantile.coords['DEPTH'].values
    taille_quantile = ds_quantile['quantile'].shape[0] 
    fig, ax = plt.subplots()  # Utilisation de plt.subplots() pour créer la figure et l'axe

    for i in range(taille_quantile):
        ds_quantile = st.session_state.ds_py.copy(deep=True)
        ds_quantile = ds_quantile.isel(quantile=i)
        ds_quantile = ds_quantile.isel(pcm_class=0)
        ax.plot(ds_quantile['PSAL_Q'].values, depth_quantile,label=ds_quantile['quantile'].values)

    ax.legend()
    ax.set_xlabel('PSAL')
    ax.set_ylabel('Profondeur')
    ax.set_title(f'Quantile PSAL')
    st.pyplot(fig)

        









if __name__ == "__main__":
    main()
