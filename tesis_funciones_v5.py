"""
NOTAS DE LA VERSIÓN 5:
- Esta versión incluye las funciones de la versión 4 y el código de 'tesis_data_poo_1.py'
  para tener todo en un solo lugar.
- Se añadieron las funciones para tratamiento de datos del notebook de las predicciones.
- Se añadieron comentarios y se reorganizaron las funciones para facilitar la lectura.
"""

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

import typing as tp
import pandas as pd
import numpy as np
import dcor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint #ENGLE-GRANGER

def color_complementario(hex_code: str)-> str:
    """
    Toma el código hexadecimal de un color
    y devuelve el código hexadecimal del color complementario.
    """

    hex_code = hex_code.lstrip('#')

    # Convierte el código hex a componentes RGB
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)

    # Calcula el complementario
    comp_r = 255 - r
    comp_g = 255 - g
    comp_b = 255 - b

    # Convierte de nuevo a formato hexadecimal
    return "#{:02X}{:02X}{:02X}".format(comp_r, comp_g, comp_b)

#-----------------#
#--COINTEGRACIÓN--#
#-----------------#

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

#------------------------#
#--TRATAMIENTO DE DATOS--#
#------------------------#

#def str_to_date(s: pd.Series) -> pd.Series:
  """
Toma una serie con índices de tipo string y la convierte a tipo datetime
  """
  s.index = pd.to_datetime(s.index)
  return s

#def quitar_findes(s: pd.Series) -> pd.Series:
  """
  Toma una serie de tiempo "s", convierte los índices a fechas
  y quita los fines de semana. Devuelve la serie sin fines de semana
  """
  #Convertimos los índices a fechas
  s = str_to_date(s)
  #Quitamos los fines de semana
  s = s[s.index.weekday < 5]
  return s

#def cut(s: pd.Series, start_date:str, end_date:str) -> pd.Series:
  """
  Toma una serie de tiempo "s" y dos cadenas de texto "start_date"
  y "end_date" en formato 'AAAA-MM-DD', devuelve la serie
  recortada para que inicie y termine en las fecha indicadas.
  """
  s = s.loc[start_date:end_date]
  return s

#def same_dates(Series: list, referencia: pd.Series) -> list:
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

#--------------------#
#--TRANSFORMACIONES--#
#--------------------#

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

#-----------------#
#--CORRELACIONES--#
#-----------------#

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

#-------------------#
#--BOLLINGER BANDS--#
#-------------------#

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

#----------------#
#--PREDICCIONES--#
#----------------#

def full_scaler(data: pd.Series, separate: bool = False, pct: float = 0.9, feature_range:tuple  = (0, 1)):
  """
  Toma una serie de tiempo, la divide y escala la primera parte, 
  usa el mismo escalador para la segunda parte.
  
  PARÁMETROS: 
  data: La serie de tiempo en cuestió

  separate: Si False (por defecto) devuelve una serie con la longitud original en una
  pandas.Series, si True, devuelve ambas partes por separado en una tupla de dos numpy.array.

  pct: Porcentaje de la serie que representa la primera parte.

  feature_range: Rango entre el que se escala la primera parte.
  """
  train_len = int(pct*len(data))
  test_len = len(data) - train_len
  train_data = data.iloc[:train_len].values
  test_data = data.iloc[train_len:].values

  scaler = MinMaxScaler(feature_range= feature_range)
  scaled_train = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
  scaled_test = scaler.transform(test_data.reshape(-1, 1)).flatten()

  scaled_data = pd.Series(np.concatenate((scaled_train, scaled_test)), name = data.name, index = data.index)

  return (scaled_train, scaled_test) if separate else scaled_data

def train_test(data: np.ndarray, p: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Toma un conjunto de datos (data) y un parámetro (p)  entre 0 y 1.
    Divide los datos, tomando el 100p% de los datos para
    entrenamiento y el resto para prueba.
    Devuelve una tupla con dos arrays, (train y test).
    """
    train = data[:int(len(data)*p)] #datos desde el 0 hasta el (100p)%
    test = data[int(len(data)*p):] #resto de los datos
    return (train,test)

def scaler(train_data , test_data, feature_range:tuple  = (0, 1)):

  scaler = MinMaxScaler(feature_range= feature_range)
  scaled_train = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
  scaled_test = scaler.transform(test_data.reshape(-1, 1)).flatten()
  return (scaled_train, scaled_test)

def split_sequence(sequence: np.ndarray, n_steps: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Divide un array de datos en subsecuencias de tamaño n_steps.
    Devuelve un array con las subsecuencias y otro con los  datos siguientes de cada subsecuencia en la original.
    """
    X, y = list(), list() #define listas vacías
    for i in range(len(sequence) - n_steps):
         # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        #el ciclo for se rompe cuando el índice final de la i-ésima
        #subsecuencia es igual o más grande que el índice final de la sec. original
        if end_ix >= len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return(np.array(X), np.array(y))

#--------------------------------------#
#-- DEFINIMOS UNA CLASE POR COMODIDAD--#
#--------------------------------------#

class TimeSeries:

    """
    Es una clase pensada para hacer gráficas de las series de tiempo, así como acceder rápidamente
    a sus rendimientos normales, logarítmicos e incluso reescalarlos con MimMaxScaler.
    ATRIBUTOS:
    .name: String con el nombre de la serie
    .color: String con el color que la serie tendrá al graficarla
    .data: Pandas Series con los datos.
    """

    def __init__(self, data:pd.Series, name:str, color:str):
        self.name = name
        self.color = color
        self.data  = data
        self.data.name = name

    # ---------------------------------------
    # Métodos de limpieza / preprocesamiento
    # ---------------------------------------

    def str_to_date(self) -> None:
        """Convierte los índices de .data a tipo datetime (modifica in place)."""
        self.data.index = pd.to_datetime(self.data.index)

    def quitar_findes(self) -> pd.Series:
        """Quita sábados y domingos de .data (modifica in place)."""
        self.str_to_date()
        self.data = self.data[self.data.index.weekday < 5]
        return self.data

    def cut(self, start_date: str, end_date: str) -> pd.Series:
        """Recorta .data  entre dos fechas (inclusive)."""
        self.data = self.data.loc[start_date:end_date]
        return self.data

    # --------------------------
    # Métodos de transformación
    # --------------------------


    def minmax(self, feature_range = (0, 1)) -> pd.Series:
      """
      Reescala .data entre el rango especificado (por defecto (0, 1)).
      Devuelve una pandas.Series con los datos reescalados.
      """
      return minmax(self.data, feature_range)

    def nreturns(self, scaled:bool = False, feature_range = (0, 1)) -> pd.Series:
      """
      Calcula los rendimientos normales de .data, si scaled = True reescala los rendimientos
      entre el rango especificado (por defecto (0, 1)).
      Devuelve una pandas.Series con los rendimientos.
      """
      nret = nreturns(self.data) if not scaled else minmax(nreturns(self.data))
      return nret


    def logreturns(self, scaled:bool = False, feature_range = (0, 1)) -> pd.Series:
      """
      Calcula los rendimientos logarítmicos de .data, si scaled = True reescala los rendimientos
      entre el rango especificado (por defecto (0, 1)).
      Devuelve una pandas.Series con los rendimientos.
      """

      nret = logreturns(self.data) if not scaled else minmax(logreturns(self.data))

      return nret



def same_dates(Series: list[TimeSeries], referencia: pd.Series) -> list:
  """
  Toma una lista  (Series) de TimeSeries, y una pandas.Series de referencia.
  Busca índices en todos los elementos de Series que no estén en referencia,
  remueve de cada elemento de Series los valores con todos esos índices.
  Retorna la lista modificada (todos los elementos tienen los mismos índices).

  """
  FALTANTES = []
  for I in Series:
    #En la lista vacía guardamos los días que están las criptos pero no en los indices
    FALTANTES.append(referencia.index.difference(I.data.index))
    #Hacemos una lista con todos los días que no están en por lo menos uno de los indices
  UFALTANTES = list(set(date for dif in FALTANTES for date in dif))

  for i in range(len(Series)):
    for date in UFALTANTES:
      Series[i].data = Series[i].data.drop(date, errors='ignore')
  return Series

def make_df(series_list: list[TimeSeries], index, trans: str = None, **kwargs) -> pd.DataFrame:
    """
    Organiza una lista de objetos TimeSeries en un DataFrame,
    aplicando el método `trans` si se especifica.

    Parámetros:
    - series_list: lista de objetos TimeSeries
    - trans: nombre del método (string) como 'nreturns', 'logreturns', etc.
    - kwargs: argumentos adicionales para el método

    Retorna:
    - DataFrame con cada serie como una fila
    """
    precios = {}

    for ts in series_list:
        if trans:
            metodo = getattr(ts, trans)
            serie = metodo(**kwargs)
        else:
            serie = ts.data

        precios[serie.name] = serie

    return pd.DataFrame(precios, index = index).T


#------------------------------------------------#
#--TRATAMOS LOS DATOS Y GUARDAMOS EN DATAFRAMES--#
#------------------------------------------------#

start_date = '2017-11-09'
end_date = '2025-02-07'
rutas = ["Nikkei-2014-2025.csv", "SP500-2014-2025.csv", "DAX40-2014-2025.csv", "FTSE250-2014-2025.csv",
         "MEX-2014-2025.csv", "BTC-2014-2025.csv", "ETH-2014-2025.csv", "XRP-2014-2025.csv",
         "Doge-2014-2025.csv", "Tether-2014-2025.csv"]
nombres = ["NIKKEI", "SP500", "DAX40", "FTSE250", "MEX", "BTC", "ETHEREUM", "XRP", "DOGE", "TETHER"]
colores = ['cyan', 'blue', 'teal', 'purple', 'royalblue', 'red', 'orangered', 'gold',  'crimson', 'darkorange']
precios_time_series = []

for i in range(len(rutas)):
  data = pd.read_csv(rutas[i]).iloc[0, 1:]
  ts = TimeSeries(data, nombres[i], colores[i])
  ts.data = ts.quitar_findes()
  ts.data = ts.cut(start_date, end_date)
  precios_time_series.append(ts)

precios_time_series = same_dates(precios_time_series, precios_time_series[5].data)
listed_nreturns = [TimeSeries(ts.nreturns(), ts.name, ts.color) for ts in precios_time_series]
listed_logreturns = [TimeSeries(ts.logreturns(), ts.name, ts.color) for ts in precios_time_series]
ref_time_series = precios_time_series[0]

Precios = make_df(precios_time_series, index = precios_time_series[0].data.index)
N_returns = make_df(listed_nreturns, index = listed_nreturns[0].data.index)
Log_returns = make_df(listed_logreturns, index = listed_nreturns[0].data.index)
Minmax = make_df(precios_time_series,index = precios_time_series[0].data.index, trans = "minmax")

def plot_the_TimeSeries(list_of_ts: list[TimeSeries], title: str = "Series de tiempo", figsize: tuple = (25, 6), xticks:list = ref_time_series.data.index[::20]):
  
  plt.figure(figsize=(25, 6))
  plt.title(title)
  for ts in list_of_ts:
    plt.plot(ts.data, label = ts.name, color = ts.color)
  plt.xticks(xticks, rotation = 90)
  plt.legend()
  plt.grid()
  plt.show()
  return


dict_colores = {
    'cyan': '#00FFFF',
    'blue': '#0000FF',
    'teal': '#008080',
    'purple': '#800080',
    'royalblue': '#4169E1',
    'red': '#FF0000',
    'orangered': '#FF4500',
    'gold': '#FFD700',
    'crimson': '#DC143C',
    'darkorange': '#FF8C00'
}

colores_complementarios = [color_complementario(dict_colores[color]) for color in colores]

