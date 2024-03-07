"""
from netCDF4 import Dataset

# Ouvrir le fichier
nc_file = Dataset('/home/mona/Téléchargements/argo_sample.nc', 'r')  

#recupere toutes les variables

for var_name in nc_file.variables:
    variable = nc_file.variables[var_name]
    
    # Accédez aux valeurs de la variable
    variable_values = variable[:]
    
    # Accédez aux valeurs valid_min et valid_max s'ils existent
    try:
        valid_min = variable.getncattr('valid_min')
        valid_max = variable.getncattr('valid_max')
    except AttributeError:
        valid_min = None
        valid_max = None
    
    # Vous pouvez traiter ou afficher les données ici en fonction de vos besoins
    print(f"Variable : {var_name}")
    print(f"Valid Min : {valid_min}")
    print(f"Valid Max : {valid_max}")

#affiche les valeurs des variables dans le fichier
depth_data = nc_file.variables['DEPTH']  
depth_values = depth_data[:]
lat_data = nc_file.variables['LATITUDE']  
lat_values = lat_data[:]
lon_data = nc_file.variables['LONGITUDE']  
lon_values = lon_data[:]
ti_data = nc_file.variables['TIME']  
ti_values = ti_data[:]
db_data = nc_file.variables['DBINDEX']  
db_values = db_data[:]
tem_data = nc_file.variables['TEMP']  
tem_values = tem_data[:]
sal_data = nc_file.variables['PSAL']  
sal_values = sal_data[:]
sig_data = nc_file.variables['SIG0']  
sig_values = sig_data[:]
br_data = nc_file.variables['BRV2']  
br_values = br_data[:]
print("depth")
print(depth_values)
print("latitude")
print(lat_values)
print("longitude")
print(lon_values)
print("time")
print(ti_values)
print("dbindex")
print(db_values)
print("temperature")
print(tem_values)
print("salinité")
print(sal_values)
print("sig0")
print(sig_values)
print("brv2")
print(br_values)


#ferme le fichier
nc_file.close()
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# Charger un jeu de données pour démonstration
iris = datasets.load_iris()
X = iris.data  # Les caractéristiques
y = iris.target  # Les étiquettes

# Réduire la dimensionnalité à 2 dimensions avec PCA
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# Créer le graphique en 2D avec des lignes
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

plt.figure(figsize=(8, 6))
for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.plot(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
             label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


    """ 

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

    """








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