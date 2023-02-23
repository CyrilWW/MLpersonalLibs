# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:54:05 2022

@author: Cyril G.
"""
import numpy as np
import pandas as pd
import itertools


def combinated_splited_list(lilist, no_empty_list=False):
    """Splite une liste en autant de couples de sous-listes que possible.
    Ex : (a,b,c) -> [[[], ['c', 'a', 'b']],
                     [['c'], ['a', 'b']],
                     [['a'], ['c', 'b']],
                     [['b'], ['c', 'a']],
                     [['c', 'a'], ['b']],
                     [['c', 'b'], ['a']],
                     [['a', 'b'], ['c']],
                     [['c', 'a', 'b'], []]]
    Si no_empty_list==True : (a,b,c) -> 
                    [[['c'], ['a', 'b']],
                     [['a'], ['c', 'b']],
                     [['b'], ['c', 'a']],
                     [['c', 'a'], ['b']],
                     [['c', 'b'], ['a']],
                     [['a', 'b'], ['c']]]

    Parameters
    ----------
    lilist : list
        Liste à spliter.
    no_empty_list : bool, optional
        Si no_empty_list==True, aucune sous-liste du split retourné ne sera vide.
    
    Returns
    -------
    splits : list
        Liste des couples.
    """
    lilist = set(lilist)
    n = len(lilist)
    splits = []
    for p in range(n+1):
        combs = itertools.combinations(lilist,p)
        for c in combs:
            list1 = list(set(c))
            list2 = list(set(lilist) - set(list1))
            if no_empty_list and (len(list1)==0 or len(list2)==0):
                pass # rien
            else:
                splits.append([list1,list2])
    return splits


def get_filtered_df(df, key_cols, key_vals, attr_cols):
    """Filtre un DataFrame selon une liste de colonne et de valeurs associées. 
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame à filtrer.
    key_cols : list
        Liste de colonnes sur lesquelles filtrer.
    key_vals : list
        Liste de valeurs sur lesquelles filtrer.
    attr_cols : list
        Liste des colonnes à retourner.

    Returns
    -------
    df2 : pandas.DataFrame
        DataFrame filtré.
    """
    df2 = df.loc[ (df[key_cols[0]]==key_vals[0]) & (df[key_cols[1]]==key_vals[1]) ][attr_cols]
    return df2


def isNaN(a):
    """Teste si une variable vaut NaN.
    Utilise l'astuce a != a, car NaN != NaN vaut True. 
    """
    return a != a


def fusionne_tab(vals):
    """Fusionne les informations d'un DataFrame, selon les lignes.
    
    Parameters
    ----------
    vals : list
        Liste de valeurs à fusionner.

    Returns
    -------
    vals2 : list
        Liste des valeurs fusionnées.
    """
    vals2 = []
    for row in vals:
        if vals2:
            for ind, val in enumerate(row):
                val2 = vals2[ind]
                # Si une info existe mais est NaN
                if isNaN(val2):
                    vals2[ind] = val # on remplace, même si c'est aussi un NaN
                # Si une info existe mais est plus courte (moins de ',')
                elif isinstance(val2,str) and isinstance(val,str):
                    if len(val2.split(',')) < len(val.split(',')):
                        # On remplace par la nouvelle valeur
                        vals2[ind] = val
                # Une info existe mais elle plus longue
                elif isinstance(val2,(float,int)) and isinstance(val,(float,int)) :
                    if val2 < val:
                        # On remplace par la nouvelle valeur
                        vals2[ind] = val
                # Une info existe mais elle plus longue
                else:
                    # On ne fait rien
                    pass

        else: # pas encore d'info
            # On remplace par la nouvelle valeur
            vals2 = list(row.copy())
            
    vals2 = np.array(vals2, dtype='object')
    return vals2


def fusion_doublons_numpy(df, key_cols, attr_cols):
    """Fusionne les informations des lignes doublons d'un DataFrame.
    
    Version basée du Numpy, plus rapide que basée sur pandas.
    On suppose que :
     - df est le dataframe contenant tous les doublons (mais si quelconque, cela marche aussi).
     - key_cols a 2 éléments
    TODO : nombre d'éléments quelconque dans key_cols.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame à fusionner.
    key_cols : list
        Liste des colonnes sur lesquelles les lignes seront fusionnées.
    attr_cols : list
        Liste des colonnes renvoyées.

    Returns
    -------
    df2 : pandas.DataFrame
        DataFrame fusionné.
    """
    # Init
    df2 = df.copy()
    
    # Détermination des clés uniques
    dico = {}
    tab_keys = df2[key_cols].values
    for tk in list(tab_keys):
        dico[tuple(tk)] = 1
    unique_tab_keys = list(dico.keys())
    
    dico = {}
    for tab_keys in unique_tab_keys:
        vals = get_filtered_df(df2,key_cols,tab_keys,attr_cols).values
        dico[tab_keys] = vals
    
    data = []
    for tab_keys in dico:
        vals = dico[tab_keys]
        vals2 = fusionne_tab(vals)
        row = list(tab_keys)
        row.extend(vals2)
        data.append(row)
    
    all_cols = key_cols.copy()
    all_cols.extend(attr_cols)
    df_fus = pd.DataFrame(data=data, columns=all_cols)
    return df_fus


def filtre_df_with_nan(df0, feat_cols):
    """
    Sépare un dataframe en deux :
        - un dataframe sans NaN dans les colonnes feat_cols
        - un dataframe avec au moins un NaN dans les colonnes feat_cols
    """
    no_nan = (df0[feat_cols].isna().sum(axis=1)==0)
    featnan = ~no_nan
    
    df_nonan = df0.loc[no_nan].copy()
    df_nan   = df0.loc[featnan].copy()
    return df_nonan, df_nan


def format_long_string(string, n_max, sep, warning=True, elude=False):
    """
    Formatte une chaine `string` (a priori trop) longue en 
     - lui ajoutant des '\n' 
     - tous les `n_max` caractères
     - mais seulement après un séparateur `sep`.
    Si un morceau est plus long que `n_max`, un warning est affiché si `warning`
    vaut `True`.
    Exemple :
     format_long_string('Cette trèèèèèèèèèèèèèèèèès longue chaine devient cette chaine recoupée',20,' ')
     ->
     'Cette \n
      trèèèèèèèèèèèèèèèèès \n
      longue chaine \n
      devient cette chaine \n
      recoupée'
    """
    list1 = string.split(sep)
    if elude and len(list1)>12:
        list1 = [list1[0], list1[1], '...', list1[-1]]
    lines = ''
    new_line = ''
    sep2 = sep
    for ind, elem in enumerate(list1):
        if ind == len(list1)-1: sep2 = ''
        # print(f"\n-----\nlines = {lines}")
        # print(f"elem  = {elem}")
        if len(elem + sep2)>n_max:
            # print("on rentre ici 1")
            # if lines:
            lines = lines + new_line + '\n'
            new_line = elem + sep2
            lines = lines + new_line + '\n'
            new_line = ''
            if warning :
                print(f"WARNING: le mot {elem} est plus long que n_max ({n_max}).")
            # print(f"new_line = {new_line}")
        else:
            # print("on rentre ici 2")
            if len(new_line + elem + sep2)<=n_max:
                # print("on rentre ici 2.1")
                new_line = new_line + elem + sep2
            else:
                # print("on rentre ici 2.2")
                lines = lines + new_line + '\n'
                new_line = elem + sep2
            # print(f"new_line = {new_line}")
    
    # Un dernier coup              
    lines = lines + new_line
    # print("-----------------------\n"+lines)
    return lines

