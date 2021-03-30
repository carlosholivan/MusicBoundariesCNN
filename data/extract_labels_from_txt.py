"""
SCRIPT NÚMERO: 0
Este script lee todas las etiquetas diferentes de todos ficheros y las mete en 
una lista. La función toma un .txt de entrada de etiquetas y devuelve una lista
con  las etiquetas limpias sin espacios, numeros...
"""

from os import listdir, walk, getcwd, sep
import numpy as np

def ReadNumbersFromLine(linea):
    linea = ''.join(i for i in linea if not i.isalpha())
    sinbarras = ''.join(j for j in linea if not j == '/')
    sinespacios = ''.join(n for n in sinbarras if not n.isspace())
    sinnombres =''.join(k for k in sinespacios if not k == '_')
    numeroslinea =''.join(p for p in sinnombres if not p == '-')
    n = ''.join(m for m in numeroslinea if not m == "'")
    numeros = ''.join(p for p in n if not p == '&')
    return numeros

def ReadDataFromtxt(directorio, archivo):
    numbers = []
    for path in listdir(directorio):
        file = open(directorio + archivo, "r")
    for line in file:
        numbers.append(ReadNumbersFromLine(line))
    file.close()
    return numbers





