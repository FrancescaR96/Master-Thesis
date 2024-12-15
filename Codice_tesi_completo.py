
@author: Francesca
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import calendar
import skill_metrics as sm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import xarray as xr


#MAPPA STAZIONI METEO SELEZIONATE 

# Carica il file CSV in un DataFrame
df = pd.read_excel('representative_stations_final.xlsx')

# salvo le coordinate e l'altitudine in array
lon = df['longitude'].to_numpy()
lat = df['latitude'].to_numpy()
elev =df['elevation'].to_numpy()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, 
                     projection= ccrs.Mercator())

#definisco colormap
cmap = plt.get_cmap('terrain')

sc=ax.scatter(df['longitude'], df['latitude'], c=elev, cmap=cmap,edgecolor='black', marker='o', s=50, transform=ccrs.PlateCarree(), zorder=11)
ax.set_extent([6, 19, 36, 48], crs=ccrs.PlateCarree())

url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'
image = cimgt.GoogleTiles(url=url)
ax.add_image(image, 6, interpolation='spline36', alpha=1)
        
ax.coastlines(zorder=10)

# Carico lo shapefile
shapefile_path = 'C:/Users/HP/Desktop/Tesi/ITA_adm/ITA_adm1.shp'
gdf = gpd.read_file(shapefile_path)

# Verifica la proiezione del GeoDataFrame dell'Italia e riproiettilo se necessario
if gdf.crs != ax.projection:
   gdf = gdf.to_crs(ax.projection.proj4_init)

gdf.plot(ax=ax, color='none', edgecolor='black')
 
# Aggiungi una barra dei colori
cbar = plt.colorbar(sc)
cbar.set_ticks([0, 500, 1000, 1500, 2000, 2500])
cbar.set_label('Altitude [m]', rotation=270, labelpad=15)

plt.show()

#PERIODI DI OSSERVAZIONE DELLE STAZIONI
df= pd.read_parquet("representative_stations_data_final.parquet")

# Converti la colonna 'data' in formato datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Controlla se ci sono valori NaN nel dataframe
if df.isnull().values.any():
    print("Il dataframe contiene valori NaN.")
else:
    print("Il dataframe non contiene valori NaN.")

# Identifica esattamente le posizioni dei valori NaN
nan_positions = df.isna().stack()
print(nan_positions[nan_positions].index.tolist())

# Lista delle variabili meteorologiche e dei codici
variables = ['t_med', 't_max', 't_min', 'rain']
codes = [
    'abr034', 'abr047', 'bsl008', 'clb002', 'cmp015', 'ero001', 'ero002', 
    'ero009', 'fvg009', 'laz013', 'laz026', 'laz033', 'lig009', 'lmb002', 
    'lmb015', 'lmb021', 'lmb039', 'lmb084', 'lmb255', 'mbr001', 'mbr006', 
    'mbr026', 'mrc009', 'pgl005', 'pgl008', 'pgl012', 'pgl040', 'pmn016', 
    'pmn033', 'pmn036', 'pmn074', 'scl007', 'scl014', 'scl073', 'srd013', 
    'trn009', 'trn017', 'tsc001', 'tsc002', 'tsc003', 'vda006', 'vnt023', 
    'vnt160'
]

# Dizionario per salvare i DataFrame filtrati
filtered_dfs = {}

# Ciclo per processare le variabili
for variable in variables:
    # Filtra il DataFrame per la variabile corrente
    df_var = df[['observation_date', 'doy', 'code', variable]].dropna()
    
    # Ordina il DataFrame
    sorted_df_var = df_var.sort_values(by=['code', 'observation_date'])
    
    # Ciclo per processare i codici
    for code in codes:
        # Filtra per codice specifico
        filtered_dfs[f"{variable}_{code}"] = sorted_df_var[sorted_df_var['code'] == code]

def find_contiguous_periods_grouper(df):
    """
    Finds periods of contiguous observations without gaps using a grouper column.

    Args:
        df: The pandas dataframe containing daily temperature data.

    Returns:
        A DataFrame with start and end dates, and duration of each contiguous period.
    """
    # Ensure the 'date' column is of type datetime64[ns]
    df['observation_date'] = pd.to_datetime(df['observation_date'])

    # Find consecutive gaps exceeding 1 day
    gaps = df['observation_date'].diff().dt.days.gt(1)
    # Create a grouper column based on gaps
    grouper = gaps.cumsum()

    # Group by grouper and get min and max date for each group
    result_df = df.groupby(grouper).agg(
        start_date=('observation_date', 'first'),
        end_date=('observation_date', 'last')
    ).reset_index()

    # Calculate duration as the difference between end and start dates
    result_df['duration'] = (result_df['end_date'] - result_df['start_date']).dt.days + 1

    # Calculate the gap size
    result_df['gap_size'] = ((result_df['start_date'].shift(-1) - result_df['end_date']).dt.total_seconds() / (60 * 60 * 24)) - 1
    result_df['gap_size'] = result_df['gap_size'].replace(np.nan, 0)
    
    return result_df

# Applicazione della funzione dinamicamente ai dataset in filtered_dfs
results = {}

for key, df in filtered_dfs.items():
    # Applica la funzione ai DataFrame filtrati
    results[key] = find_contiguous_periods_grouper(df)


# Inizializzare le liste per i giorni di osservazione e i gap per ogni variabile
somme_durata = []
somme_gap = []
variabili = ['rain', 't_med', 't_max', 't_min']

# Ciclo per ogni variabile
for variabile in variabili:
    # Ciclo per ogni stazione
    for stazione in codes:
        # Genera la chiave per accedere al DataFrame
        key = f'{variabile}_{stazione}'
        
        # Controlla se la chiave esiste nel dizionario results
        if key in results:
            # Ottieni il DataFrame per la variabile e stazione corrente
            df_stazione = results[key]
            
            # Calcola la somma delle durate e dei gap per la stazione corrente
            somma_durate = df_stazione['duration'].sum()
            somma_gap = df_stazione['gap_size'].sum()
            
            # Aggiungi i valori alle liste
            somme_durata.append(somma_durate)
            somme_gap.append(somma_gap)
        else:
            print(f'La chiave {key} non è presente in results.')

# Crea un nuovo DataFrame per le somme totali
data_somme = {'Stazione': codes * len(variabili),  # 4 variabili per ogni stazione
              'Variabile': variabili * len(codes),
              'Somma Durata': somme_durata,
              'Somma Gap': somme_gap}

df_somme = pd.DataFrame(data_somme)


# Funzione per calcolare il numero totale di giorni in un anno, considerando se è bisestile o meno
def giorni_nell_anno(anno):
    if calendar.isleap(anno):
        return 366
    else:
        return 365

# Funzione per calcolare il numero totale di giorni in un mese, considerando i giorni del mese
def giorni_nel_mese(anno, mese):
    # Lista dei giorni per ogni mese (31 giorni per Gennaio, Marzo, Maggio, etc., 30 per Aprile, Giugno, etc.)
    giorni_per_mese = {1: 31, 2: 29 if calendar.isleap(anno) else 28, 3: 31, 4: 30, 5: 31, 6: 30, 
                       7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    return giorni_per_mese[mese]

# Lista per raccogliere i risultati
somme_anni = []
somme_mesi = []


# Ciclo per ogni variabile
for variabile in ['rain', 't_med', 't_max', 't_min']:
    # Ciclo per ogni stazione
    for stazione in codes:
        # Genera la chiave per accedere al DataFrame
        key = f'{variabile}_{stazione}'
        
        # Controlla se la chiave esiste nel dizionario results (o filtered_dfs)
        if key in filtered_dfs:
            # Ottieni il DataFrame per la variabile e stazione corrente
            df_stazione = filtered_dfs[key]
            
            # Converte la colonna 'observation_date' in datetime se non lo è già
            if not pd.api.types.is_datetime64_any_dtype(df_stazione['observation_date']):
                df_stazione['observation_date'] = pd.to_datetime(df_stazione['observation_date'], errors='coerce')
            
            # Rimuove le righe con valori NaT (valori di data non validi)
            df_stazione = df_stazione.dropna(subset=['observation_date'])
            
            # Reimposta gli indici per evitare problemi successivi
            df_stazione = df_stazione.reset_index(drop=True)
            
            # Estraiamo l'anno e il mese dalla colonna 'observation_date'
            df_stazione['anno'] = df_stazione['observation_date'].dt.year
            df_stazione['mese'] = df_stazione['observation_date'].dt.month
            
            # Calcolo delle osservazioni per ogni anno
            osservazioni_per_anno = df_stazione.groupby(['anno'])['observation_date'].nunique().reset_index(name='osservazioni_presenti_anno')
            osservazioni_per_anno['giorni_totali_anno'] = osservazioni_per_anno['anno'].apply(giorni_nell_anno)
            osservazioni_per_anno['Osservazioni Mancanti Anno'] = osservazioni_per_anno['giorni_totali_anno'] - osservazioni_per_anno['osservazioni_presenti_anno']
            
            # Aggiungi colonne per variabile e stazione
            osservazioni_per_anno['variabile'] = variabile
            osservazioni_per_anno['stazione'] = stazione
            
            # Calcolo delle osservazioni per ogni mese
            osservazioni_per_mese = df_stazione.groupby(['anno', 'mese'])['observation_date'].nunique().reset_index(name='osservazioni_presenti_mese')
            osservazioni_per_mese['giorni_totali_mese'] = osservazioni_per_mese.apply(lambda row: giorni_nel_mese(row['anno'], row['mese']), axis=1)
            osservazioni_per_mese['Osservazioni Mancanti Mese'] = osservazioni_per_mese['giorni_totali_mese'] - osservazioni_per_mese['osservazioni_presenti_mese']
            
            # Aggiungi colonne per variabile e stazione
            osservazioni_per_mese['variabile'] = variabile
            osservazioni_per_mese['stazione'] = stazione
            
            # Aggiungi i risultati alle liste
            somme_anni.append(osservazioni_per_anno)
            somme_mesi.append(osservazioni_per_mese)
            
# Concatenamento dei risultati per tutte le variabili e stazioni
df_somme_anni = pd.concat(somme_anni).reset_index(drop=True)
df_somme_mesi = pd.concat(somme_mesi).reset_index(drop=True)

# media dei giorni in un anno
mean_days_in_year = 365.25

def calcola_anni(df, mean_days):
    df['Anni_osservazioni_tot']= df['Somma Durata']/ mean_days
    df['Anni_mancanti_tot']= df['Somma Gap']/ mean_days
    #approssimazioni
    df['Anni_tot_approx']= df['Anni_osservazioni_tot'].round(1)
    df['Anni_mancanti_tot_approx']= df['Anni_mancanti_tot'].round(1)
    return df

# Applica calcoli in base alla variabile
df_somme = df_somme.groupby('Variabile', group_keys=False).apply(lambda df: calcola_anni(df, mean_days_in_year))     

#grafico a barre
def plot_observation_periods(observation_periods, df_somme, variable_name, bar_color):
    """
    Plots observation periods as horizontal bars with gaps for a given variable.
    
    Parameters:
    - observation_periods: dict, mapping station names to lists of observation periods
    - df_somme: DataFrame, contains total observation years for each station
    - variable_name: str, name of the variable being analyzed (e.g., 't_med', 't_max')
    - bar_color: str, color of the bars in the plot
    """
    # Creare il grafico a barre
    fig, ax = plt.subplots(figsize=(15, 10))

    for i, (station, periods) in enumerate(observation_periods.items()):
        # Recupera il numero di osservazioni per la stazione
        numero_osservazioni = df_somme[df_somme['Stazione'] == station]['Anni_tot_approx'].values[0]
        
        # Aggiungere le barre per ciascun periodo di osservazione
        for period in periods:
            start_date, end_date = period
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
            
            ax.broken_barh(
                [(mdates.date2num(start_datetime), 
                  mdates.date2num(end_datetime) - mdates.date2num(start_datetime))],
                (i - 0.4, 0.8),
                facecolors=bar_color,
                edgecolor='black',
                zorder=2
            )

        # Posiziona l'etichetta solo alla fine dell'ultima barra  
        ax.text(mdates.date2num(end_datetime) + 30, i, str(numero_osservazioni),
                color='black', fontsize=11, ha='left', va='center')

    # Configura assi e etichette
    ax.set_ylim(-1, len(observation_periods))
    ax.set_yticks(range(len(observation_periods)))
    ax.set_yticklabels(observation_periods.keys())
    ax.set_xlim(mdates.date2num(datetime(2001, 1, 1)), mdates.date2num(datetime(2024, 12, 31)))
    ax.set_xlabel('Years')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, zorder=1)
    ax.set_ylabel('Observation Station', fontsize=12)

    # Aggiungere didascalia per i numeri alla fine delle barre (verticale)
    ax.text(1.01, 0.5, 'Total Observations (years)', color='black', fontsize=12, 
            ha='left', va='center', rotation='vertical', transform=ax.transAxes)

    # Titolo personalizzato per la variabile
    plt.title(f'Observation periods with gaps - {variable_name.capitalize()}')
    plt.tight_layout()
    plt.show()

observation_periods = {
    'vnt160' :[('2010-11-06','2022-10-18')],
    'vnt023' :[('2005-06-11','2023-10-08')],
    'vda006' :[('2013-11-26','2023-10-08')],
    'tsc003' :[('2002-11-07','2006-06-1'), ('2006-08-12','2010-05-19'), ('2010-07-27','2023-04-11')],
    'tsc002' :[('2002-11-16','2023-10-08')],
    'tsc001' :[('2002-11-11','2023-10-08')],
    'trn017' :[('2007-04-03','2007-05-03'), ('2007-10-23','2007-11-11'), ('2008-02-18','2012-10-14'), ('2013-06-04','2023-10-08')],
    'trn009' :[('2005-12-20','2006-04-19'), ('2007-05-10','2023-10-08')],
    'srd013' :[('2010-03-04','2012-07-13'), ('2012-09-05','2013-09-22'), ('2014-05-18','2014-08-30'), ('2015-04-15','2022-08-17'), ('2022-10-09','2023-06-02'), ('2023-07-28', '2023-10-08')],
    'scl073' :[('2011-11-15','2022-07-17'), ('2022-09-21','2023-10-08')],
    'scl014' :[('2005-05-16','2010-06-24'), ('2010-09-03','2011-08-11'), ('2011-09-19','2013-12-05'), ('2014-01-2','2023-10-08')],
    'scl007' :[('2004-12-01','2023-10-08')],
    'pmn074' :[('2011-08-24','2020-02-19'), ('2020-03-26','2023-10-19')],
    'pmn036' :[('2007-02-06','2023-10-08')],
    'pmn033' :[('2006-09-23','2023-10-08')],
    'pmn016' :[('2003-12-29','2017-06-02'), ('2017-10-01','2023-10-08')],
    'pgl040' :[('2009-10-24','2009-11-05'), ('2010-02-12','2010-09-20'), ('2010-10-20','2022-11-28'), ('2023-01-08','2023-10-10')],
    'pgl012' :[('2007-10-14','2009-01-09'), ('2009-02-04','2013-01-08'), ('2014-01-23','2018-11-04'), ('2019-07-13','2023-10-10')],
    'pgl008' :[('2005-05-16','2006-09-20'), ('2007-01-01','2010-07-29'), ('2010-08-29','2012-01-16'), ('2012-02-18','2012-04-28'), ('2012-06-06','2017-11-15'), ('2018-01-04','2020-07-06'), ('2020-09-19','2021-01-04'), ('2021-03-29','2022-06-07'), ('2023-09-12','2023-10-08')],
    'pgl005' :[('2005-01-21','2006-09-12'), ('2007-10-05','2008-03-30'), ('2009-04-15','2023-10-08')],
    'mrc009' :[('2005-10-27','2005-12-12'), ('2006-06-01','2020-04-30')],
    'mbr026' :[('2012-07-10','2022-10-15')],
    'mbr006' :[('2007-05-11','2023-10-10')],
    'mbr001' :[('2003-10-23','2004-03-26'), ('2009-06-18','2011-02-24'), ('2011-07-06','2016-09-23'), ('2017-01-26','2018-09-27'), ('2018-11-10','2023-10-08')],
    'lmb255' :[('2011-07-05','2017-08-04'), ('2017-11-07','2023-10-20')],
    'lmb084' :[('2004-12-05','2023-10-19')],
    'lmb039' :[('2002-11-23','2004-05-28'), ('2004-09-27','2006-04-13'), ('2007-04-01','2007-05-24'), ('2007-07-29','2023-10-08')],
    'lmb021' :[('2001-01-01','2023-10-20')],
    'lmb015' :[('2002-11-01','2002-11-19'), ('2003-06-15','2004-10-01'), ('2004-11-22','2013-08-18'), ('2014-09-07','2022-05-17')],
    'lmb002' :[('2002-11-11','2002-11-21'), ('2003-07-01','2023-10-08')],
    'lig009' :[('2008-10-09','2010-02-04'), ('2010-03-15','2022-06-22')],
    'laz033' :[('2008-03-02','2008-10-19'), ('2008-11-16','2023-10-08')],
    'laz026' :[('2006-11-13','2006-11-13'), ('2009-01-04','2009-07-24'), ('2009-08-27','2018-10-28'), ('2018-12-23','2023-10-10')],
    'laz013' :[('2006-02-23','2010-11-24'), ('2011-03-30','2023-10-08')],
    'fvg009' :[('2007-11-13','2023-10-08')],
    'ero009' :[('2003-08-26','2010-06-21'), ('2010-07-26','2011-07-24'), ('2011-09-04','2012-07-23'), ('2012-08-24','2017-05-15'),('2017-06-14','2020-02-14')],
    'ero002' :[('2002-11-13','2002-12-12'), ('2003-02-06','2003-05-05'), ('2004-10-10','2023-10-08')],
    'ero001' :[('2005-05-15','2007-06-02'), ('2007-07-09','2010-03-31'), ('2010-10-26','2023-09-29')],
    'cmp015' :[('2009-05-11','2012-03-11'), ('2012-08-23','2023-10-08')],
    'clb002' :[('2007-04-01','2017-11-12'), ('2018-12-06','2021-11-24'), ('2022-01-02','2022-05-24'), ('2022-08-31','2022-09-26'), ('2022-10-27','2023-10-08')],
    'bsl008' :[('2008-03-29','2011-07-30'), ('2011-09-13','2023-10-08')],
    'abr047' :[('2012-11-05','2023-10-10')],
    'abr034' :[('2011-05-29','2023-10-08')]
}

df_somme_tmed = df_somme[df_somme['Variabile'] == 't_med']
df_somme_tmax = df_somme[df_somme['Variabile'] == 't_max']
df_somme_tmin = df_somme[df_somme['Variabile'] == 't_min']
df_somme_p = df_somme[df_somme['Variabile'] == 'rain']

plot_observation_periods(observation_periods, df_somme_tmed, 't_med', 'tab:blue')

#MERGE DATA
earth_radius = 6371.0

stations = pd.read_excel('representative_stations_final.xlsx')
data_stations = pd.read_parquet("representative_stations_data_final.parquet")

def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * earth_radius

model ='era5_land'
points = 1 # how many grid points
var = 't_med'

results = []


for code in stations['code'].unique():
    df_model = pd.read_parquet(f"{code}_{model}.parquet")
    # Split point string to recover lat/lon coordinates
    df_model[['lat','lon']] = df_model['point'].str.split("_", expand=True).rename(columns={0:'lat',1:'lon'}).astype(float)
    # Compute distance between grid point and station (in km)
    df_model['distance'] = haversine(df_model['lon'], df_model['lat'], # Grid points
                               *stations.query(f"code == '{code}'")[['longitude','latitude']].values[0] # Station coordinates
                               )
 # Filter to subset only a number of points
    # We first order the points by increasing distance
    distances = df_model[['point','distance']].drop_duplicates().sort_values(by='distance')
    # And then select only the closest N points
    sel_points = distances.iloc[0:points]
    df_model = df_model[df_model.point.isin(sel_points['point'])]
    #
    print(f"Station={code}, model={model}, n_points={len(df_model.point.unique())} -> Max. distance = {sel_points['distance'].max():.2f}")
    # Now take the data from the station
    df = data_stations.query(f"code == '{code}'").copy()
    # Select only the variable we need and set index
    df = df[['observation_date', var]].set_index('observation_date').dropna()
    # Merge the two dataframes to consider the common period
    output = df.merge(df_model[['time',var]], left_index=True, right_on='time',
                      suffixes=('_station','_model')).set_index('time').sort_index()
    output['code'] = code
    # Append to the results list
    results.append(output)
    
# Concatenate the results into a single dataframe
results = pd.concat(results)

#statistiche per Taylor plot
statistics= sm.taylor_statistics(results[f'{var}_model'], results[f'{var}_station'])

#diagramma di Taylor
sdev = np.array([
    statistics_tmed_era5land['sdev'][0], statistics_tmed_era5land['sdev'][1], 
    statistics_tmed_cerra['sdev'][1], statistics_tmed_merida['sdev'][1],
    statistics_tmed_mswx['sdev'][1], statistics_tmed_vhr['sdev'][1]
])

crmsd = np.array([
    statistics_tmed_era5land['crmsd'][0], statistics_tmed_era5land['crmsd'][1], 
    statistics_tmed_cerra['crmsd'][1], statistics_tmed_merida['crmsd'][1],
    statistics_tmed_mswx['crmsd'][1], statistics_tmed_vhr['crmsd'][1]
])

ccoef = np.array([
    statistics_tmed_era5land['ccoef'][0], statistics_tmed_era5land['ccoef'][1], 
    statistics_tmed_cerra['ccoef'][1], statistics_tmed_merida['ccoef'][1],
    statistics_tmed_mswx['ccoef'][1], statistics_tmed_vhr['ccoef'][1]
])

plt.figure(figsize=(10, 6))
sm.taylor_diagram(
                  sdev,
                 crmsd,
                 ccoef,
                 styleOBS = '-',
                 colOBS = '#AAAADD',
                 markerobs = 'o',
                 titleOBS = 'observation',
                 markerLabel = ['OBS', 'ERA5-Land', 'CERRA-Land', 'MERIDA', 'MSWX' ,'VHR-REA_IT', 'MSWEP', 'CHIRPS'],
                 markerLabelColor = 'r',
                 markerLegend = 'on',
                 markerSize = 10,
                 colframe = '#DDDDDD',
                 colscor = {'grid': '#DDDDDD','tick_labels': '#000000','title': '#000000'},
                 colsstd = {'grid': '#DDDDDD','tick_labels': '#000000','ticks': '#DDDDDD','title': '#000000'},
                 colRMS = '#DDDDDD',
                 labelweight = 'normal',
                 showlabelsRMS='off',
                 tickRMSangle=140,
                 numberPanels = 1)
            
plt.gca().set_title('T_med n=1', fontsize=15, loc='center')
plt.show()


# calcolo bias
# Lista per memorizzare le medie dei bias per ogni stazione
station_biases = []

for code in stations['code'].unique():
    # Seleziona i risultati per la stazione corrente
    output = results[results['code'] == code].copy()

    # Calcola il bias per ogni punto e in ogni momento
    output['bias'] = output[var + '_model'] - output[var + '_station']

    # Calcola la media del bias per la stazione
    station_bias = output.groupby('code')['bias'].mean().reset_index()
    station_biases.append(station_bias)

# Concatena le medie dei bias in un unico dataframe
average_biases_df = pd.concat(station_biases)


# La funzione shift_colormap per mantenre scala di coloro centrata nello zero 

def shift_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'): 
    if isinstance(cmap, str): 
        cmap = cm.get_cmap(cmap) 

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []} 

    # Regular index to compute the colors 
    reg_index = np.linspace(start, stop, 257) 

    # Shifted index to match the data 
    shift_index = np.hstack([ 
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True) 

    ]) 

    for ri, si in zip(reg_index, shift_index): 
        r, g, b, a = cmap(ri) 
        cdict['red'].append((si, r, r)) 
        cdict['green'].append((si, g, g)) 
        cdict['blue'].append((si, b, b)) 
        cdict['alpha'].append((si, a, a)) 

    new_cmap = mcolors.LinearSegmentedColormap(name, cdict) 
    cm.register_cmap(name, new_cmap) 
    return new_cmap 

# Imposta i limiti della colormap 
vmin = -8 
vmax = 8 
# Calcola il midpoint per la colormap shiftata 
midpoint = 1 - vmax / (vmax + abs(vmin)) 

# Usa un nome unico per la colormap 
unique_cmap_name = 'shiftedcmap_custom' 

# Crea la colormap personalizzata solo se non esiste già 
if unique_cmap_name not in plt.colormaps(): 
    cmap = shift_colormap('seismic', midpoint=midpoint, name=unique_cmap_name) 
else: 
    cmap = plt.get_cmap(unique_cmap_name) 

# Utilizza Normalize per saturare i valori al di fuori dei limiti 
norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True) 

# Funzione per creare la mappa 
def plot_bias_map(lon, lat, bias, title, shapefile_path): 
    fig = plt.figure(figsize=(10, 10)) 
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator()) 

    # Usa la colormap personalizzata, imposta vmin e vmax 
    sc = ax.scatter(lon, lat, c=bias, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=80, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree(), zorder=11) 
    ax.set_extent([6, 19, 36, 48], crs=ccrs.PlateCarree()) 

    url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg' 
    image = cimgt.GoogleTiles(url=url) 
    ax.add_image(image, 6, interpolation='spline36', alpha=1) 

    ax.coastlines(zorder=10) 

    gdf = gpd.read_file(shapefile_path) 

    if gdf.crs != ax.projection: 
        gdf = gdf.to_crs(ax.projection.proj4_init) 

    gdf.plot(ax=ax, color='none', edgecolor='black') 


    # Aggiungi una barra dei colori 
    cbar = plt.colorbar(sc, label='Bias') 
    cbar.set_ticks(np.arange(vmin, vmax+1, 2)) 
    cbar.set_label('Bias [°C]', rotation=270, labelpad=15) 

    plt.title(title) 
    plt.show() 

  
#mappa dei bias 
lon = stations['longitude'].to_numpy() 
lat = stations['latitude'].to_numpy() 
bias = average_biases_df['bias'].to_numpy() 

# Plotta la mappa 
plot_bias_map(lon, lat, bias, 'Mean error (Model-Obs)', 'C:/Users/HP/Desktop/Tesi/ITA_adm/ITA_adm1.shp') 

  
#IDW
model ='era5_land'
points = 16 # how many grid points
var = 'rain'

results = []

for code in stations['code'].unique():
    df_model = pd.read_parquet(f"{code}_{model}.parquet")
    # Split point string to recover lat/lon coordinates
    df_model[['lat','lon']] = df_model['point'].str.split("_", expand=True).rename(columns={0:'lat',1:'lon'}).astype(float)
    # Compute distance between grid point and station (in km)
    df_model['distance'] = haversine(df_model['lon'], df_model['lat'], # Grid points
                               *stations.query(f"code == '{code}'")[['longitude','latitude']].values[0] # Station coordinates
                               )
    # Calculate weights based only on distance
    df_model['weight'] = 1 / (pow(df_model['distance'],2) + 1)
 # Filter to subset only a number of points
    # We first order the points by increasing distance
    distances = df_model[['point','distance']].drop_duplicates().sort_values(by='distance')
    # And then select only the closest N points
    sel_points = distances.iloc[0:points]
    df_model = df_model[df_model.point.isin(sel_points['point'])]
    #
    print(f"Station={code}, model={model}, n_points={len(df_model.point.unique())} -> Max. distance = {sel_points['distance'].max():.2f}")
    # Now take the data from the station
    df = data_stations.query(f"code == '{code}'").copy()
    # Select only the variable we need and set index
    df = df[['observation_date', var]].set_index('observation_date').dropna()
    # Merge the two dataframes to consider the common period
    output = df.merge(df_model[['time',var, 'weight']], left_index=True, right_on='time',
                      suffixes=('_station','_model')).set_index('time').sort_index()
    output['code'] = code
    # Append to the results list
    results.append(output)
    
# Concatenate the results into a single dataframe
results_rain_era5land = pd.concat(results)

# Group by 'time' and 'code' and calculate weighted mean of 'rain_model'
weighted_mean = results_rain_era5land.groupby(['time', 'code']).apply(lambda x: np.average(x['rain_model'], weights=x['weight'])).reset_index(name='rain_model_weighted')

# Take the first value of 'rain_station' for each combination of 'time' and 'code'
rain_station_first = results_rain_era5land.groupby(['time', 'code'])['rain_station'].first().reset_index()

# Merge 'weighted_mean' with 'rain_station_first' based on 'time' and 'code'
results_rain_era5land_weighted = pd.merge(weighted_mean, rain_station_first, on=['time', 'code'], how='left')

# Sort the final dataset first by 'code' and then by 'time'
results_rain_era5land_weighted_sorted = results_rain_era5land_weighted.sort_values(by=['code', 'time'])

statistics_rain_era5land= sm.taylor_statistics(results_rain_era5land_weighted_sorted[f'{var}_model_weighted'], results_rain_era5land_weighted_sorted[f'{var}_station'])

#CORREZIONE QUOTA
dataset_cerra = xr.open_dataset('C:/Tesi/climatologies/oro_lsm_cerra_italy.nc', engine='netcdf4')
dataset_era5land = xr.open_dataset('C:/Tesi/climatologies/era5_land_oro_it.nc', engine='netcdf4')
dataset_merida = xr.open_dataset('C:/Tesi/climatologies/inv_hres.nc', engine='netcdf4')

print(dataset_cerra)
print(dataset_cerra.data_vars)
print(dataset_cerra['orog'])

print(dataset_era5land)
print(dataset_era5land.data_vars)
print(dataset_era5land['h'])

print(dataset_merida)
print(dataset_merida.data_vars)


df_orog_cerra = dataset_cerra['orog'].to_dataframe()
df_orog_era5land = dataset_era5land['h'].to_dataframe()
df_orog_merida = dataset_merida['HGT'].to_dataframe()


df_orog_cerra = df_orog_cerra[['latitude', 'longitude', 'orog']]

df_orog_era5land.reset_index(inplace=True)
df_orog_era5land = df_orog_era5land[['latitude', 'longitude', 'h']]

df_orog_merida.reset_index(inplace=True)
df_orog_merida = df_orog_merida[['lat', 'lon', 'HGT']]

earth_radius = 6371.0

stations = pd.read_excel('representative_stations_final.xlsx')
data_stations = pd.read_parquet("representative_stations_data_final.parquet")

def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * earth_radius


model = 'cerra'
points = 4  # Numero di punti della griglia
var = 't_med'

results = []

for code in stations['code'].unique():
    df_model = pd.read_parquet(f"{code}_{model}.parquet")
    # Split point string per recuperare le coordinate lat/lon
    df_model[['lat', 'lon']] = df_model['point'].str.split("_", expand=True).rename(columns={0: 'lat', 1: 'lon'}).astype(float)
    # Calcola la distanza tra il punto della griglia e la stazione (in km)
    df_model['distance'] = haversine(df_model['lon'], df_model['lat'],  # Punti della griglia
                                      *stations.query(f"code == '{code}'")[['longitude', 'latitude']].values[0]  # Coordinate della stazione
                                      )
    # Calculate weights based only on distance
    df_model['weight'] = 1 / (pow(df_model['distance'],2) + 1)
    # Filtra per selezionare solo un numero di punti
    # Ordina i punti per distanza crescente
    distances = df_model[['point', 'distance']].drop_duplicates().sort_values(by='distance')
    # Seleziona solo i primi N punti più vicini
    sel_points = distances.iloc[0:points]
    df_model = df_model[df_model.point.isin(sel_points['point'])]

    print(f"Stazione={code}, modello={model}, n_punti={len(df_model.point.unique())} -> Distanza massima = {sel_points['distance'].max():.2f}")

    # Seleziona i dati dalla stazione
    df = data_stations.query(f"code == '{code}'").copy()
    # Seleziona solo la variabile necessaria e imposta l'indice
    df = df[['observation_date', var]].set_index('observation_date').dropna()
    # Unisci i due dataframes per considerare il periodo comune
    output = df.merge(df_model[['time', var,'weight', 'lat', 'lon', 'distance']], left_index=True, right_on='time',
                      suffixes=('_station', '_model')).set_index('time').sort_index()
    output['code'] = code
    results.append(output)

# Concatena i risultati in un singolo dataframe
results_tmed_cerra = pd.concat(results)

results_tmed_cerra.reset_index(inplace=True)

#approssimo colonne per poi fare merge dei dataset
df_orog_cerra['latitude'] = df_orog_cerra['latitude'].round(2)
df_orog_cerra['longitude'] = df_orog_cerra['longitude'].round(2)

# Effettua il merge
merged_df_tmed_cerra = results_tmed_cerra.merge(df_orog_cerra, left_on=['lat', 'lon'], right_on=['latitude', 'longitude'])

# Rimuovi le colonne duplicate (latitude e longitude)
merged_df_tmed_cerra.drop(['latitude', 'longitude'], axis=1, inplace=True)


# Definisci una funzione per calcolare la differenza tra l'elevazione della stazione e l'altitudine del punto della griglia
def elevation_difference(station_elev, grid_elev):
    return station_elev - grid_elev

# Inizializza lista vuota per contenere i valori della nuova colonna
elevation_differences = []

# Itera attraverso le righe di merged_df
for index, row in merged_df_tmed_cerra.iterrows():
    # Estrai il codice della stazione corrente
    station_code = row['code']
    
    # Trova le informazioni sulla stazione corrente nel dataframe delle stazioni
    station_info = stations.loc[stations['code'] == station_code]
    
    # Estrai l'altitudine della stazione corrente
    station_elev = station_info['elevation'].values[0]
    
    # Calcola la differenza di elevazione tra la stazione e il punto della griglia
    elev_difference = elevation_difference(station_elev, row['orog'])
    elevation_differences.append(elev_difference)

# Aggiungi la nuova colonna a merged_df
merged_df_tmed_cerra['elevation_difference'] = elevation_differences

# Ordina il DataFrame per il codice e per la data
merged_df_tmed_cerra = merged_df_tmed_cerra.sort_values(by=['code', 'time'])

#regeressione per calcolare lapse rate
from sklearn.linear_model import LinearRegression

# Inizializza una lista per salvare i coefficienti angolari (lapse rate)
lapse_rates_data = []

# Raggruppa il dataframe per 'code' (stazione) e 'time' (data)
grouped = merged_df_tmed_cerra.groupby(['code', 'time'])

# Itera su ciascun gruppo
for (code, time_tmed_cerra), group_data in grouped:
    # Estrai le feature e il target per la regressione
    X = group_data[['orog']]
    y = group_data[f'{var}_model']
    
    # Inizializza il modello di regressione lineare
    model = LinearRegression()
    
    # Addestra il modello
    model.fit(X, y)
    
    # Ottieni il coefficiente angolare della regressione
    lapse_rate = model.coef_[0]
    
    # Aggiungi il coefficiente angolare alla lista lapse_rates_data ripetendolo per ogni riga nel gruppo
    lapse_rates_data.extend([{'code': code, 'time': time_tmed_cerra, 'lapse_rate': lapse_rate}] * len(group_data))

# Converti la lista in un dataframe
lapse_rates = pd.DataFrame(lapse_rates_data)

#calcolo delta t quota
merged_df_tmed_cerra['deltat_quota'] = lapse_rates['lapse_rate'] * merged_df_tmed_cerra['elevation_difference']

# Creazione della nuova colonna "t_med_corrected" inizialmente uguale a "t_med_model"
merged_df_tmed_cerra[f'{var}_corrected'] = merged_df_tmed_cerra[f'{var}_model']

# Applica la correzione del t_med_corrected in base al valore di deltat_quota
merged_df_tmed_cerra[f'{var}_corrected'] -= np.where(
    merged_df_tmed_cerra['elevation_difference'] > 0,  # Se l'altitudine della stazione è maggiore
    merged_df_tmed_cerra['deltat_quota'],             # Sottrai deltat_quota
    -merged_df_tmed_cerra['deltat_quota']             # Aggiungi deltat_quota
)

# Group by 'time' and 'code' and calculate weighted mean of t_med_corrected
weighted_mean = merged_df_tmed_cerra.groupby(['time', 'code']).apply(lambda x: np.average(x[f'{var}_corrected'], weights=x['weight'])).reset_index(name=f'{var}_corrected_weighted')

# Take the first value of 'tmed_station' for each combination of 'time' and 'code'
t_med_station_first = merged_df_tmed_cerra.groupby(['time', 'code'])[f'{var}_station'].first().reset_index()

# Merge 'weighted_mean' with 'tmed_station_first' based on 'time' and 'code'
results_tmed_cerra_weighted = pd.merge(weighted_mean, t_med_station_first, on=['time', 'code'], how='left')

# Sort the final dataset first by 'code' and then by 'time'
results_tmed_cerra_weighted_sorted = results_tmed_cerra_weighted.sort_values(by=['code', 'time'])

statistics_tmed_cerra= sm.taylor_statistics(results_tmed_cerra_weighted_sorted[f'{var}_corrected_weighted'], results_tmed_cerra_weighted_sorted[f'{var}_station'])

#MACHINE LEARNING
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import matplotlib.pyplot as plt


stations = pd.read_excel("representative_stations_final.xlsx")
stations.head()

data = pd.read_parquet("representative_stations_data_final.parquet")
data.head()


code = 'tsc001'

station = stations[stations.code==code][['code', 'latitude', 'longitude',
                'elevation']].to_dict('records')[0]

df = data.loc[data.code==code, ['t_min', 't_med', 't_max', 'rain', 'observation_date','doy']].copy()
df.head()

reanalysis = 'era5_land' # era5_land, cerra, merida_hres, mswx, vhr, mswep, chirps

df_nearest = pd.read_parquet(f"{station['code']}_{reanalysis}.parquet")
df_nearest.head()

df_nearest.tail()

# Contare il numero di punti per latitudine e longitudine
point_counts = df_nearest.groupby(['point']).size().reset_index(name='counts')
print(point_counts)

var = 't_med'

df_nearest_cp = df_nearest.pivot_table(
    values=var, index='time', columns='point').copy()
input_data = df[['observation_date', var]].set_index('observation_date')
input_data = input_data.merge(
    df_nearest_cp, left_index=True, right_index=True)
input_data = input_data.dropna()

X = input_data.loc[:, input_data.columns != var].values
Y = input_data.loc[:, input_data.columns == var].values.ravel()

#SVR
#hyperparametrization
param_grid = {'C': np.linspace(1, 5, 2),
              'gamma': np.linspace(1E-6, 1E-1, 2),
              'kernel': ['linear', 'rbf'],
              'epsilon':np.linspace(1E-3, 1E-1, 2)}

scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
grid = GridSearchCV(SVR(), param_grid, scoring=scoring, refit='r2', verbose=2, n_jobs=4, return_train_score=True)
grid.fit(X, Y)


#best score è la media degli score ottenuto da diversi dataset usati durande cross-validazione
print("Best parameters found:", grid.best_params_)
print("Best score:", grid.best_score_)


# Calcolare il RMSE medio e il R² medio per i dati di training e validation
results = pd.DataFrame(grid.cv_results_)

# RMSE medio in calibrazione (training)
mean_train_mse = -results['mean_train_mse'].iloc[grid.best_index_]
# RMSE medio in validazione (validation)
mean_valid_mse = -results['mean_test_mse'].iloc[grid.best_index_]

# RMSE medio in calibrazione (training)
mean_train_rmse = np.sqrt(mean_train_mse)
# RMSE medio in validazione (validation)
mean_valid_rmse = np.sqrt(mean_valid_mse)

# R² medio in calibrazione (training)
mean_train_r2 = results['mean_train_r2'].iloc[grid.best_index_]
# R² medio in validazione (validation)
mean_valid_r2 = results['mean_test_r2'].iloc[grid.best_index_]

print(f'RMSE medio in calibrazione: {mean_train_rmse}')
print(f'RMSE medio in validazione: {mean_valid_rmse}')
print(f'R² medio in calibrazione: {mean_train_r2}')
print(f'R² medio in validazione: {mean_valid_r2}')

#salva il modello migliore
joblib.dump(grid.best_estimator_, f'best_svr_model_{code}_{reanalysis}_{var}.pkl')


#Random Forest
# Hyperparametrizzazione
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [ 5, 10],
    'min_samples_leaf': [ 2, 4],
    'bootstrap': [True, False]
}

scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring=scoring, refit='r2', verbose=2, n_jobs=4, return_train_score=True)
grid.fit(X, Y)

print("Best parameters found:", grid.best_params_)
print("Best score:", grid.best_score_)

# Calcolare il RMSE medio e il R² medio per i dati di training e validation
results = pd.DataFrame(grid.cv_results_)

# RMSE medio in calibrazione (training)
mean_train_mse = -results['mean_train_mse'].iloc[grid.best_index_]
# RMSE medio in validazione (validation)
mean_valid_mse = -results['mean_test_mse'].iloc[grid.best_index_]

# RMSE medio in calibrazione (training)
mean_train_rmse = np.sqrt(mean_train_mse)
# RMSE medio in validazione (validation)
mean_valid_rmse = np.sqrt(mean_valid_mse)

# R² medio in calibrazione (training)
mean_train_r2 = results['mean_train_r2'].iloc[grid.best_index_]
# R² medio in validazione (validation)
mean_valid_r2 = results['mean_test_r2'].iloc[grid.best_index_]

print(f'RMSE medio in calibrazione: {mean_train_rmse}')
print(f'RMSE medio in validazione: {mean_valid_rmse}')
print(f'R² medio in calibrazione: {mean_train_r2}')
print(f'R² medio in validazione: {mean_valid_r2}')

# Salva il modello migliore
joblib.dump(grid.best_estimator_, f'best_rf_model_{code}_{reanalysis}_{var}.pkl')

#precipitazioni
# Optional weights for rainy days
sample_weights = None
rainy_weight = 1e4
non_rainy_weight = 0.0
sample_weights = np.where(Y > 0, rainy_weight, non_rainy_weight)

#SVR
#hyperparametrization
param_grid = {'C': np.linspace(0.01, 0.1, 2),
              'gamma': np.linspace(1E-6, 1E-1, 2),
              'kernel': ['rbf'],
              'epsilon':np.linspace(1E-3, 1E-1, 2)}

scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
grid = GridSearchCV(SVR(), param_grid, scoring=scoring, refit='r2', verbose=2, n_jobs=4, return_train_score=True)
grid.fit(X, Y, sample_weight=sample_weights)


#best score è la media degli score ottenuto da diversi dataset usati durande cross-validazione
print("Best parameters found:", grid.best_params_)
print("Best score:", grid.best_score_)


# Calcolare il RMSE medio e il R² medio per i dati di training e validation
results = pd.DataFrame(grid.cv_results_)

# MSE medio in calibrazione (training)
mean_train_mse = -results['mean_train_mse'].iloc[grid.best_index_]
# MSE medio in validazione (validation)
mean_valid_mse = -results['mean_test_mse'].iloc[grid.best_index_]

# RMSE medio in calibrazione (training)
mean_train_rmse = np.sqrt(mean_train_mse)
# RMSE medio in validazione (validation)
mean_valid_rmse = np.sqrt(mean_valid_mse)

# R² medio in calibrazione (training)
mean_train_r2 = results['mean_train_r2'].iloc[grid.best_index_]
# R² medio in validazione (validation)
mean_valid_r2 = results['mean_test_r2'].iloc[grid.best_index_]

print(f'RMSE medio in calibrazione: {mean_train_rmse}')
print(f'RMSE medio in validazione: {mean_valid_rmse}')
print(f'R² medio in calibrazione: {mean_train_r2}')
print(f'R² medio in validazione: {mean_valid_r2}')

# Salvataggio del modello migliore
joblib.dump(grid.best_estimator_, f'best_svr_model_{code}_{reanalysis}_{var}_kernel_rbf.pkl')

#RICOSTRUZIONE CLIMATOLOGIE
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_data(code, stations, data, cerra_file, mswep_file):
    station = stations[stations.code == code][['code', 'latitude', 'longitude', 'elevation']].to_dict('records')[0]
    df = data.loc[data.code == code, ['t_min', 't_med', 't_max', 'rain', 'observation_date', 'doy']].copy()

    df_cerra = pd.read_parquet(cerra_file)
    df_mswep = pd.read_parquet(mswep_file)
    
    return df, df_cerra, df_mswep, station

def predict_temperatures(df_cerra, best_models, temperature_variables):
    results_monthly_temp = {}
    for var in temperature_variables:
        df_nearest_cp = df_cerra.pivot_table(values=var, index='time', columns='point').dropna()
        X_to_predict = df_nearest_cp.loc['1991':'2020'].values
        predicted_values = best_models[var].predict(X_to_predict)
        
        results_daily = pd.DataFrame(index=df_nearest_cp.loc['1991':'2020'].index, data={var: predicted_values})
        results_daily['month'] = results_daily.index.month
        results_monthly_temp[var] = results_daily.groupby('month').mean()

    return results_monthly_temp

def predict_precipitation(df_mswep, best_model):
    df_nearest_cp = df_mswep.pivot_table(values='rain', index='time', columns='point').dropna()
    X_to_predict = df_nearest_cp.loc['1991':'2020'].values
    predicted_precipitation = best_model.predict(X_to_predict)

    results_daily = pd.DataFrame(index=df_nearest_cp.loc['1991':'2020'].index, data={'rain': predicted_precipitation})
    results_daily['month'] = results_daily.index.month
    results_daily['year'] = results_daily.index.year
    results_monthly_precip = results_daily.groupby(['year', 'month']). sum().reset_index()

    # Calcolo della media mensile cumulata per la precipitazione
    monthly_precipitation_cumulative = results_monthly_precip.groupby(['month'])['rain'].mean().reset_index()
    return monthly_precipitation_cumulative['rain'].values

def plot_climatology(results_monthly_temp, monthly_precipitation_mean_values, code):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    months = np.arange(1, 13)

    # Temperature
    colors = {'t_min': 'black', 't_med': 'orange', 't_max': 'green'}
    temp_min = float('inf')
    temp_max = float('-inf')

    for var in results_monthly_temp.keys():
        ax1.plot(results_monthly_temp[var].index, results_monthly_temp[var][var], label=f'{var}', color=colors[var], marker='o')
        temp_min = min(temp_min, results_monthly_temp[var][var].min())
        temp_max = max(temp_max, results_monthly_temp[var][var].max())

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'Monthly Climatology of Reconstructed Series (1991-2020) - Station {code}')
    ax1.grid(True, zorder=11)

   # Imposta i limiti dell'asse delle temperature
    lower_limit = -5 if temp_min < 0 else 0  # Imposta il limite inferiore a -5 se c'è un valore negativo
    upper_limit = temp_max + 5  # Imposta il limite superiore con un margine di +5 rispetto al massimo
    ax1.set_ylim(lower_limit, upper_limit)

    # Forza i tick dell'asse delle temperature a includere numeri negativi se necessario
    temp_ticks = np.arange(lower_limit, upper_limit + 1, 5)  # Tick ogni 5 gradi, regolando con i limiti inferiori e superiori
    ax1.set_yticks(temp_ticks)
    
    # Precipitazione
    ax2 = ax1.twinx()
    ax2.bar(months, monthly_precipitation_mean_values, color='b', alpha=0.3, label='Rain')
    ax2.set_xticks(months)
    ax2.set_xticklabels(months)

    # Imposta dinamicamente i limiti dell'asse delle precipitazioni
    max_precipitation = monthly_precipitation_mean_values.max()
    upper_limit = max_precipitation * 1.2
    ax2.set_ylim(0, upper_limit)

    # Imposta i tick dell'asse y delle precipitazioni in modo che corrispondano alle linee della griglia
    yticks = np.linspace(0, upper_limit)
    ax2.set_ylabel('Precipitation (mm)')

    # Legenda
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

# Carica i dati
stations = pd.read_excel("representative_stations_final.xlsx")
data = pd.read_parquet("representative_stations_data_final.parquet")

# Codice della stazione
code = 'abr034'

df, df_cerra, df_mswep, station = load_data(code, stations, data, f"{code}_cerra.parquet", f"{code}_mswep.parquet")

# Modelli già calibrati e caricati
best_models = {
    't_max': joblib.load("best_svr_model_abr034_cerra_t_max_kernel_linear.pkl"),
    't_min': joblib.load("best_svr_model_abr034_cerra_t_min_kernel_linear.pkl"),
    't_med': joblib.load("best_svr_model_abr034_cerra_t_med_kernel_linear.pkl"),
    'rain': joblib.load("best_svr_model_abr034_mswep_rain_kernel_rbf.pkl")
}

# Predizione delle temperature
temperature_variables = ['t_min', 't_med', 't_max']
results_monthly_temp = predict_temperatures(df_cerra, best_models, temperature_variables)

# Predizione delle precipitazioni
monthly_precipitation_mean_values = predict_precipitation(df_mswep, best_models['rain'])

# Creazione del grafico
plot_climatology(results_monthly_temp, monthly_precipitation_mean_values, code)
