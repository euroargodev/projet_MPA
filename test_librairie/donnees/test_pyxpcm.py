# fonctionne ! 
import numpy as np

# Temporarily redefine np.int to int -> car sinon on a une erreur np.int est deprecier dans NumPy 1.20
np.int = int

# Load libraries
import numpy as np
from pyxpcm.models import pcm
import xarray as xr
from argopy import DataFetcher as ArgoDataFetcher
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import gsw
import seaborn


# créer une box pour dataFetcher de argopy
llon=-75;rlon=-45
ulat=30;llat=20
depthmin=0;depthmax=600
# Time range des donnnées
time_in='2010-01-01'
time_f='2010-12-12'


#recuperer les données avec argopy
ds_points = ArgoDataFetcher(src='erddap').region([llon,rlon, llat,ulat, depthmin, 700,time_in,time_f]).to_xarray()
#mettre en 2 dimensions ( N_PROf x N_LEVELS)
ds = ds_points.argo.point2profile()
#recuperer les données SIG0 et N2(BRV2) avec teos 10
ds.argo.teos10(['SIG0','N2'])


print(ds)

# z est créé pour représenter des profondeurs de 0 à la profondeur maximale avec un intervalle de 5 mètres ( peux etre modifié)
depthmax = depthmax + 10
z=np.arange(0,depthmax,10)
#interpole ds avec les profondeurs z
print(z)
ds2 = ds.argo.interp_std_levels(z)
print ("ici")
print(ds2)

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
print("ici")


element_depth = da.sel(DEPTH=da.DEPTH[-1]).DEPTH.values# récupère la valeur max de DEPTH
z = np.arange(0.,element_depth,-10.) # depth array
pcm_features = {'temperature': z, 'salinity':z} #features that vary in function of depth
m = pcm(K=6, features=pcm_features, maxvar=2) # create the 'basic' model
print(m)


features_in_ds = {'temperature': 'TEMP', 'salinity': 'PSAL'}
features_zdim='DEPTH'

das = da.pyxpcm.fit_predict(m, features=features_in_ds, dim=features_zdim, inplace=True)
das['TEMP'].attrs['feature_name'] = 'temperature'
das['PSAL'].attrs['feature_name'] = 'salinity'
das['DEPTH'].attrs['axis'] = 'Z'
print(das)



m.fit(da, features=features_in_ds, dim=features_zdim)
da['TEMP'].attrs['feature_name'] = 'temperature'
da['PSAL'].attrs['feature_name'] = 'salinity'
da['DEPTH'].attrs['axis'] = 'Z'


m.predict(da, features=features_in_ds, dim=features_zdim,inplace=True)
print(da)

m.predict_proba(da, features=features_in_ds, inplace=True)
print(da)

for vname in ['TEMP', 'PSAL']:
    da = da.pyxpcm.quantile(m, q=[0.05, 0.5,0.95], of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

print(da)

X, sampling_dims = m.preprocessing(das, features=features_in_ds)
print(X)

m.plot.scaler()
#m.plot.reducer()
#g = m.plot.preprocessed(das, features=features_in_ds, style='darkgrid')
#g = m.plot.preprocessed(das, features=features_in_ds, kde=True)

fig, ax = m.plot.quantile(da['TEMP_Q'], maxcols=3, figsize=(10, 8), sharey=True)
#fig, ax = m.plot.quantile(da['PSAL_Q'], maxcols=3, figsize=(10, 8), sharey=True)

plt.show()



#plt.show()

"""
proj = ccrs.PlateCarree()
subplot_kw={'projection': proj, 'extent': np.array([-80,1,-1,66]) + np.array([-0.1,+0.1,-0.1,+0.1])}
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=120, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)

kmap = m.plot.cmap()
sc = ax.scatter(da['LONGITUDE'], da['LATITUDE'], s=3, c=da['PCM_LABELS'], cmap=kmap, transform=proj, vmin=0, vmax=m.K)
cl = m.plot.colorbar(ax=ax)

gl = m.plot.latlongrid(ax, dx=10)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.set_title('LABELS of the training set')
plt.show()
"""



"""
        colum_x = st.selectbox('Sélectionner x', st.session_state.ds_py.to_dataframe().columns)
        ds_data = st.session_state.ds_py.copy(deep =True)

        mean_by_depth = ds_data.mean(dim='N_PROF')[colum_x]
        depth_values = ds_data.coords['DEPTH'].values

        fig, ax = plt.subplots()
        ax.plot(mean_by_depth,depth_values)
        ax.set_xlabel('Moyenne de {colum_x}')
        ax.set_ylabel('Profondeur')
        ax.set_title(f'Moyenne de {colum_x} par profondeur')
        ax.invert_yaxis()  # Inverser l'axe y pour avoir la profondeur croissante vers le bas
        st.pyplot(fig)
        
"""

# Reset np.int to its original value
np.int = np.int_

