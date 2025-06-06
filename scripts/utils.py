#FUNCIONES GENERALES:

import pandas as pd
import numpy as np
from scipy.stats import mode

def elegir_imputacion(df, columna, agrupacion, umbral_dispersion=60):
    """
    Evalúa la dispersión de datos dentro de grupos y selecciona si usar moda o mediana para imputar valores nulos.

    Args:
        df (pandas.DataFrame): DataFrame con datos a limpiar.
        columna (str): Nombre de la columna a imputar.
        agrupacion (list): Lista de columnas para agrupar.
        umbral_dispersion (int): Si la dispersión (IQR) supera este valor, se usa la mediana; si es menor, la moda.

    Returns:
        pandas.DataFrame: DataFrame con valores nulos reemplazados de forma adaptativa.
    """
    
    def calcular_iqr(series):
        """Calcula la dispersión IQR dentro de una serie."""
        q1, q3 = np.nanpercentile(series.dropna(), [25, 75])
        return q3 - q1
    
    def imputar_fecha(grupo):
        """Decide si usar moda o mediana según la dispersión del grupo."""
        dispersion = calcular_iqr(grupo)  # Calculamos la dispersión IQR
        if dispersion > umbral_dispersion:
            return grupo.fillna(grupo.median())  # Si la dispersión es alta, usamos la mediana
        else:
            try:
             return grupo.fillna(mode(grupo.dropna())[0][0])  # Si es baja, usamos la moda
            except IndexError:  # Manejo de error si no hay moda válida
             return grupo.fillna(grupo.median())
    
    # Aplicamos la imputación dentro de cada grupo
    df[columna] = df.groupby(agrupacion)[columna].transform(imputar_fecha)
    
    return df


def cargar_datos(file_path):
    """
    Carga un archivo CSV en un DataFrame de Pandas.

    Args:
        file_path (str): Ruta del archivo CSV.

    Returns:
        pandas.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(file_path)

def estandarizar_nombres_columnas(df):
    """
    Estandariza los nombres de las columnas: convierte a minúsculas y reemplaza espacios con guiones bajos.

    Args:
        df (pandas.DataFrame): DataFrame con columnas sin estandarizar.

    Returns:
        pandas.DataFrame: DataFrame con nombres de columnas estandarizados.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def convertir_a_fecha(df, columnas):
    """
    Convierte las columnas especificadas a formato datetime.

    Args:
        df (pandas.DataFrame): DataFrame con datos sin formato de fecha.
        columnas (list): Lista de nombres de columnas a convertir.

    Returns:
        pandas.DataFrame: DataFrame con columnas en formato datetime.
    """
    for columna in columnas:
        df[columna] = pd.to_datetime(df[columna], errors='coerce')
    return df

def limpiar_texto(df, columnas, conservar_mayusculas=[]):
    """
    Elimina espacios extra y convierte texto a minúsculas en columnas seleccionadas,
    pero conserva las mayúsculas en las columnas especificadas.

    Args:
        df (pandas.DataFrame): DataFrame con datos sin limpiar.
        columnas (list): Lista de nombres de columnas a limpiar.
        conservar_mayusculas (list): Lista de columnas en las que no se aplicará la conversión a minúsculas.

    Returns:
        pandas.DataFrame: DataFrame con texto estandarizado.
    """
    for columna in columnas:
        if columna in df.columns and df[columna].dtype == "object":  
            df[columna] = df[columna].str.strip() 
            if columna not in conservar_mayusculas:
              df[columna] = df[columna].str.lower() 
    return df


def resumen_nulos(df):
    """
    Muestra el número y porcentaje de valores nulos por columna.

    Args:
        df (pandas.DataFrame): DataFrame a evaluar.

    Returns:
        pandas.DataFrame: Resumen con la cantidad y porcentaje de valores nulos.
    """
    resumen = df.isnull().sum().to_frame(name="cantidad_nulos")
    resumen["porcentaje_nulos"] = (resumen["cantidad_nulos"] / len(df)) * 100
    return resumen.sort_values(by="porcentaje_nulos", ascending=False)

def evaluar_fila(df, columna_nula, columnas_clave, umbral=0.7):
    """
    Evalúa si una fila con valores nulos en una columna tiene valores importantes en otras columnas clave.

    Args:
        df (pandas.DataFrame): DataFrame a evaluar.
        columna_nula (str): Columna con valores nulos que queremos analizar.
        columnas_clave (list): Lista de columnas consideradas importantes.
        umbral (float): Porcentaje mínimo de valores no nulos en las columnas clave antes de eliminar la fila (por defecto 50%).

    Returns:
        pandas.DataFrame: Filas que podrían eliminarse según la evaluación.
    """
    # Filtrar filas con valores nulos en la columna de interés
    df_nulo = df[df[columna_nula].isnull()].copy()
    
    # Contar cuántas columnas clave tienen valores no nulos en cada fila
    df_nulo["proporcion_datos_importantes"] = df_nulo[columnas_clave].notnull().sum(axis=1) / len(columnas_clave)
    
    # Filtrar filas donde el porcentaje de datos no nulos en columnas clave es bajo
    filas_a_eliminar = df_nulo[df_nulo["proporcion_datos_importantes"] < umbral]

    return filas_a_eliminar.drop(columns=["proporcion_datos_importantes"])


    
