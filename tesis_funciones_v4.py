# -*- coding: utf-8 -*-
"""
Funciones, datos y demás que uso en los 
diferentes Notebooks para la Tesis
Incluye el tratamiento de las series de tiempo:
- Que todas empiezen y terminen en la misma fecha
- Quitar los fines de semana en los índices financieros
- Estandarizar las fechas, cada país tiene sus propios bussines days
- Quitar los no bussines days a las criptomonedas
Al final las diez series deben tener la misma longitud

"""


import pandas as pd
import numpy as np
import dcor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint #ENGLE-GRANGER

def is_stationary(series: pd.Series) -> bool:
    """
    Aplica el test de Dickey-Fuller aumentado para determinar si una serie es estacionaria.
    Retorna True si la serie es estacionaria y False en caso contrario.
    """
    result = adfuller(series)[1]
    return True if result < 0.05 else False


def is_I1(serie: pd.Series) -> bool:
  """
  Determina si una serie de tiempo es I(1) o no.
  Retorna True si la serie es I(1)y False en caso contrario.
  Requiere:
  from statsmodels.tsa.stattools import adfuller
  """
  precios = is_stationary(serie)
  diferencia = is_stationary(serie.diff().dropna())
  
  return True if not precios and diferencia else False 


def are_cointegrated(series1: pd.Series, series2: pd.Series) -> bool:
    """
   Aplica el test de Engle-Granger para determinar si dos series son cointegradas.
   Retorna True si las series son cointegradas y False en caso contrario.
   Requiere:
   from statsmodels.tsa.stattools import coint
    """
    pvalue= coint(series1, series2)[1]
    return True if pvalue < 0.05 else False


def str_to_date(s: pd.Series) -> pd.Series:
  """
Toma una serie con índices de tipo string y la convierte a tipo datetime
  """
  s.index = pd.to_datetime(s.index)
  return s


def quitar_findes(s: pd.Series) -> pd.Series:
  """
  Toma una serie de tiempo "s", convierte los índices a fechas
  y quita los fines de semana. Devuelve la serie sin fines de semana
  """
  #Convertimos los índices a fechas
  s = str_to_date(s)
  #Quitamos los fines de semana
  s = s[s.index.weekday < 5]
  return s


def EG_comb_lin(xt: pd.Series, yt: pd.Series) ->  pd.Series:
  """
  Toma dos series de tiempo y devuelve
  la combinación lineal estacionaria (si la hay según el test EG).
  de la forma e_t = yt-(alpha + beta*xt)
  Requiere:
  from statsmodels.tsa.stattools import adfuller
  import statsmodels.api as sm
  from statsmodels.tsa.stattools import coint
  """
  if not are_cointegrated(xt, yt):
    print("Las series no están cointegradas en ese sentido")
    return None
  else:
    x_const = sm.add_constant(xt) #De statsmodels.api (sm)

    # Hacemos la regresión: y = alpha + beta * x + errorb
    model = sm.OLS(yt, x_const)
    results = model.fit()

    # Obtenemos los residuos = combinación lineal estacionaria
    cl = results.resid
    alpha, beta  = results.params.iloc[0], results.params.iloc[1] #(alpha, beta)
    #cl_t = pd.Series(yt.values - ( alpha + beta*xt.values), index = mt.yt.index)

    return alpha, beta, cl


def cut(s: pd.Series, start_date:str, end_date:str) -> pd.Series:
  """
  Toma una serie de tiempo "s" y dos cadenas de texto "start_date"
  y "end_date" en formato 'AAAA-MM-DD', devuelve la serie
  recortada para que inicie y termine en las fecha indicadas.
  """
  s = s.loc[start_date:end_date]
  return s


def nreturns(s: pd.Series) -> pd.Series:
  """
  Toma una serie de tiempo y devuelve una serie con los
  rendimientos normales
  """
  S = s.values.astype(np.float32)
  rendimientos = np.zeros(len(S))
  for i in range(1, len(S)):
    rendimientos[i] = (S[i] - S[i-1])/S[i-1]
  nombre_base = s.name if s.name is not None else "sin_nombre"
  return pd.Series(rendimientos[1:], index = s.index[1:], name = nombre_base)# + "_nr")



def logreturns(S: pd.Series) -> pd.Series:
  """
Toma una serie de tiempo "S" y devuelve una serie con los rendimientos
logarítmicos
  """
  valores = S.values.astype(np.float32)
  logret = np.zeros(len(valores))
  for i in range(1, len(valores)):
      logret[i] = np.log(valores[i]/valores[i-1])
  nombre_base = S.name if S.name is not None else "sin_nombre"
  return pd.Series(logret[1:], index = S.index[1:], name = nombre_base) #+ "_lr")


def minmax(s: pd.Series, feature_range = (0, 1)) -> pd.Series:
  """
  Toma una serie de tiempo y devuelve una serie con los
  rendimientos normalizados usando el método MinMaxScaler
  """
  scaler = MinMaxScaler(feature_range= feature_range)
  scaled = scaler.fit_transform(s.values.reshape(-1, 1))
  nombre_base = s.name if s.name is not None else "sin_nombre"
  return pd.Series(scaled.flatten(), index = s.index, name = nombre_base) #+ "_mm")


def DCC_matrix(M: np.array, Round = 2, **kwargs) -> np.array:
  """
  Toma una matriz con N renglones, devuelve una matriz
  NxN con los DCC (Distance Correlation Coeficient)
  entre cada una de las N series
  """
  N = M.shape[0]
  DCC_m = np.zeros((N,N))
  for i in range(0, N, 1):
    for j in range(0, N, 1):
      DCC_m[i,j] = dcor.distance_correlation(M[i][0:], M[j][0:], **kwargs)

  return np.round(DCC_m, Round)


def PCC_matrix(M: np.array, Round = 2, **kwargs) -> np.array:
   """
   Devuelde una matriz con los PCC redondeados(Perason Correlation Coeficient)
   entre las series de tiempo contenidas en M.
   """
   return np.round(np.corrcoef(M, **kwargs), 2)


def same_dates(Series: list, referencia: pd.Series) -> list:
  FALTANTES = []
  for I in Series:
    #En la lista vacía guardamos los días que están las criptos pero no en los indices
    FALTANTES.append(referencia.index.difference(I.index))
    #Hacemos una lista con todos los días que no están en por lo menos uno de los indices
  UFALTANTES = list(set(date for dif in FALTANTES for date in dif))

  for i in range(len(Series)):
    for date in UFALTANTES:
      Series[i] = Series[i].drop(date, errors='ignore')
  return Series


def Bollinger_bands(serie: pd.Series, window = 20, n=2):
    """
    Calcula las bandas de Bollinger para una serie de tiempo.
    """
    media_movil = serie.rolling(window=window).mean()
    desv_estand = serie.rolling(window=window).std()
    upper_band = media_movil + n * desv_estand
    lower_band = media_movil - n * desv_estand
    return media_movil, upper_band, lower_band


def my_plotted_bb(serie:pd.Series, title:str, bands = None,  ticks = 20, ROI: list = None, *args, **kwargs):
    
    """
    Grafica las bandas de Bollinger de una serie de tiempo.
    ROI = [start_date, end_date], 'yyyy-mm-dd'
    bands = [Moving_averge, upper_band, lower_band]
    if bands is None, reutrns a moving average and the Bollinger bands for serie
    """
    color_s = "#ED553B"
    color_ma = "#F6D55C"
    color_bu = "#3CAEA3"
    color_bl ="#20639B"

    if bands is None:
       bb= Bollinger_bands(serie, **kwargs)
       ma = bb[0]
       bu = bb[1]
       bl = bb[2]
      
    else:
      ma = bands[0]
      bu = bands[1]
      bl = bands[2]
       
    if ROI is None:

       plt.figure(figsize=(30,10))
       plt.title(title, fontsize=20)
       BU, = plt.plot(bu, color = color_bu)
       BL, = plt.plot(bl,  color = color_bl)
       CA, = plt.plot(serie, color = color_s)
       MA, =plt.plot(ma, color=color_ma)

       plt.fill_between(bu.index, bu, bl, alpha=0.15)
       plt.xticks(serie.index[::ticks], rotation=45)
       handles, labels = plt.gca().get_legend_handles_labels()

       #Reordenar los handles y labels (cambiamos el orden en que se muestran en la leyenda)
       plt.legend(handles=[CA, BU, MA, BL], labels=["Precios", "BU", "MA", "BL"], fontsize=20)
       plt.grid()
       plt.show()

    else:
       start_date = ROI[0]
       end_date = ROI[1]
       plt.figure(figsize=(30,10))
       plt.title(title, fontsize=20)
       BU, = plt.plot(bu[start_date:end_date], color = color_bu)
       BL, = plt.plot(bl[start_date:end_date], color = color_bl)
       CA, = plt.plot(serie[start_date:end_date], color = color_s)
       MA, =plt.plot(ma[start_date:end_date], color=color_ma)
      
       plt.fill_between(bu[start_date:end_date].index, bu[start_date:end_date], bl[start_date:end_date], alpha=0.15)
       plt.xticks(serie.loc[start_date:end_date].index[::ticks], rotation=45)
       handles, labels = plt.gca().get_legend_handles_labels()

       #Reordenar los handles y labels (cambiamos el orden en que se muestran en la leyenda)
       plt.legend(handles=[CA, BU, MA, BL], labels=["Precios", "BU", "MA", "BL"], fontsize=20)
       plt.grid()
       plt.show()
    return ma, bu, bl


def split_epochs(M: np.array, dt: int) -> np.array:
  """
  Toma una matriz NxT y la divide en submatrices Nxdt
  Devuelve int(t/dt) submatrices Nxdt y la matriz residual
  (si la hay)
  """
  N, t = M.shape
  n = int(t/dt)
  residuo = t - int(n*dt)
  Epocas = np.zeros((n, N, dt))
  for i in range(n):
    Epocas[i] = M[:, i*dt:(i+1)*dt]
  
  E_res = M[:, -residuo:]
  return Epocas, E_res


def split_epochs_df(M: pd.DataFrame, dt: int):
    
    """
    Toma un DataFrame con forma NxT (filas = N, columnas = T) 
    y lo divide en sub-DataFrames Nxdt.
    Devuelve una lista de n = int(T/dt) sub-DataFrames 
    y el DataFrame residual (si lo hay).
    """
    N, T = M.shape
    n = T // dt
    residuo = T - n * dt

    Epocas = []
    for i in range(n):
        sub_df = M.iloc[:, i*dt:(i+1)*dt]
        Epocas.append(sub_df)

    E_res = M.iloc[:, -residuo:] if residuo > 0 else pd.DataFrame()

    return Epocas, E_res


def make_df(Series: list[pd.Series], Trans = None, **kwargs) -> pd.DataFrame:
  
  """
  Organiza las series de una lista en un data frame
  Aplica la función Trans a las Series.
  """
  Precios = {}
  for s in Series:
     serie = Trans(s, **kwargs) if Trans else s
     Precios[serie.name] = serie
  Precios = pd.DataFrame(Precios).T
  return Precios





#TRATAMIENTO DE DATOS

sp500 = pd.read_csv("SP500-2014-2025.csv").iloc[0 ,1:]
nikkei = pd.read_csv("Nikkei-2014-2025.csv").iloc[0 ,1:]
ftse = pd.read_csv("FTSE250-2014-2025.csv").iloc[0 ,1:]
dax = pd.read_csv("DAX40-2014-2025.csv").iloc[0 ,1:]
mex = pd.read_csv("MEX-2014-2025.csv").iloc[0 ,1:]
btc = pd.read_csv("BTC-2014-2025.csv").iloc[0 ,1:]
eth = pd.read_csv("ETH-2014-2025.csv").iloc[0 ,1:]
xrp = pd.read_csv("XRP-2014-2025.csv").iloc[0 ,1:]
tether = pd.read_csv("Tether-2014-2025.csv").iloc[0 ,1:]
doge = pd.read_csv("Doge-2014-2025.csv").iloc[0 ,1:]


#NOMBRAMOS LAS SERIES
nombres = ["Nikkei", "SP500", "DAX40", "FTSE250", "MEX", "BTC", "ETHEREUM", "XRP", "DOGE", "TETHER" ]
Series = [nikkei, sp500, dax, ftse, mex, btc, eth, xrp, doge, tether]
for s, n in zip(Series, nombres):
   s.name = n

#MISMA LONGITUD Y FECHAS
start_date = '2017-11-09'
end_date = '2025-02-07'

for i in range(0, 5, 1):
    Series[i] = cut(str_to_date(Series[i]), start_date, end_date)
for i in range(5, 10, 1):
    Series[i] = cut(quitar_findes(Series[i]), start_date, end_date)

Series = same_dates(Series, referencia = btc)


#ORGANIZAMOS EN DATA FRAMES
Precios = make_df(Series)
N_returns = make_df(Series, nreturns)
Log_returns = make_df(Series, logreturns)
Minmax = make_df(Series, minmax)
  
