# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:08:03 2022

@author: a072108
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro, kstest, kruskal
import math

from DisplotData import bcolors

def influenceCatVal(featValues,importance, sep='_'):
    """
    On suppose que les catégories encodées ont un et un seul underscore dans le nom des colonnes.
    Si ce n'est pas le cas, chaque feature sans underscore sera considérée comme célibataire.
    """    
    influenceCateg = {}
    for encFeat,imp in zip(featValues,importance):
        if len(encFeat.split('_'))<2:
            cat = encFeat
        else :
            cat,val = encFeat.split(sep)
        if cat not in influenceCateg:
            influenceCateg[cat] = np.abs(imp)
        else:
            influenceCateg[cat] += np.abs(imp)

    # influenceVals = [ [k,v] for k,v in influenceCateg.items() ]
    influenceVals = influenceCateg
    # influenceVals = influenceCateg.sort(key=lambda x: x[1],reverse=True)
    # influenceVals = influenceCateg.sort(key=lambda x: x.value(),reverse=True)
    # influenceVals = dict(sorted(influenceCateg.items(), key=lambda item: item[1], reverse=True))
    influenceVals = dict(sorted(influenceCateg.items(), key=lambda item: item[1]))
    return influenceVals

def kruskal_custo(df,categ,num_cols):
    distrib_egales = False
    for var in num_cols:
        print(bcolors.BOLD+bcolors.OKGREEN+var+'\n'+'-'*len(var)+bcolors.ENDC)
        # Test du test
        # x = np.random.randn(n) # DEBUG

        H, p = kruskal(*[df.loc[df[categ]==grp,var].values for grp in df[categ].unique()])
        print(f'  H = {H:.3f}') # | seuil = {seuil:.12f}')
        print(f'  p = {p:.12f}')

        if p<0.05: # or H >= seuil:
            distrib_egales = False
            print(f"H0 rejetée, les distributions sont significativement différentes pour la catégorie '{categ}'")
        else:
            distrib_egales = True
        print('')
    return distrib_egales

def ks_custo(df_imp,categ,num_cols):
    au_moins_une_dist_normale = False
    for var in num_cols:
        for grp in df_imp[categ].unique():
            x = df_imp.loc[df_imp[categ]==grp,var].values
            n = len(x)
            # Centrage-réduction de x
            x = StandardScaler().fit_transform(x.reshape(-1, 1))

            # Test du test
            # z = np.random.randn(n) # DEBUG

            K, p = kstest(x,'norm')
            seuil = 0.886/math.sqrt(n)
            if K <= seuil or p >= 0.05:
                au_moins_une_dist_normale = True
                print(bcolors.BOLD+bcolors.OKGREEN+var+'\n'+'-'*len(var)+bcolors.ENDC)
                #print(bcolors.BOLD+' '+grp+bcolors.ENDC)
                print(bcolors.BOLD+' Enfin une distribution normale !! → '+bcolors.OKCYAN+grp+bcolors.ENDC)
                print(f' K = {K:.3f} | seuil = {seuil:.12f}')
                print(f' p = {p:.12f}')
    if not au_moins_une_dist_normale:
        s = f"Aucune distribution n'est normale selon la catégorie '{categ}'"
        print(bcolors.BOLD+bcolors.FAIL+s+'\n'+'-'*len(s)+bcolors.ENDC)
    return au_moins_une_dist_normale

def shapiro_custo(df_imp,categ,num_cols):
    au_moins_une_dist_normale = False
    for var in num_cols:
        for grp in df_imp[categ].unique():
            x = df_imp.loc[df_imp[categ]==grp,var].values
            x = StandardScaler().fit_transform(x.reshape(-1, 1))
            n = len(x)
            if n>5000:
                n=5000
            # Test du test
            # x = np.random.randn(n) # DEBUG

            W, p = shapiro(x[0:n])
            if p>=0.05:
                au_moins_une_dist_normale = True
                print(bcolors.BOLD+bcolors.OKGREEN+var+'\n'+'-'*len(var)+bcolors.ENDC)
                #print(bcolors.BOLD+' '+grp+bcolors.ENDC)
                print(bcolors.BOLD+' Enfin une distribution normale !! → '+bcolors.OKCYAN+grp+bcolors.ENDC)
                print(f'  n = {n}')
                print(f'  W = {W:.3f}')
                print(f'  p = {p:.12f}')
                print('')
    if not au_moins_une_dist_normale:
        s = f"Aucune distribution n'est normale selon la catégorie '{categ}'"
        print(bcolors.BOLD+bcolors.FAIL+s+'\n'+'-'*len(s)+bcolors.ENDC)
    return au_moins_une_dist_normale



# ------------------------------------------------------------------------------
# Fonctions OpenClassRooms (Nicolas Rangeon), merci à eux
# ------------------------------------------------------------------------------
