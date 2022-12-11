# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:54:05 2022

@author: a072108
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import seaborn as sns

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
# for color in [bcolors.HEADER,bcolors.OKBLUE,bcolors.OKCYAN,bcolors.OKGREEN,bcolors.WARNING,bcolors.FAIL,bcolors.BOLD,bcolors.UNDERLINE]:
#     print(color+f'Quelques couleurs, pour agrémenter'+bcolors.ENDC)
    
def template(a, b=''):
    """Template de fonction suivant la PEP8.
    
    Parameters
    ----------
    a : str
        Première chaine à concaténer.
    b : str, optional
        Deuxième chaine à concaténer.

    Returns
    -------
    c : str
        Chaine concaténée.
    """
    return a+b


def print_debug(ch, message_level=1, debug_level=0):
    """Affiche un message sous condition.
    
    Parameters
    ----------
    ch : str
        Chaine à afficher.
    message_level: int, optional
        Niveau intrinsèque du message.
    debug_level: int, optional
        Niveau de debug en cours.
    """
    if debug_level>=message_level:
        print(ch)


def applique_style_paragraphe_1(style,titre,texte=''):
    """Applique un style "paragraphe" à un couple titre/texte.
    
    Le titre est coloré mais pas le texte.

    Parameters
    ----------
    style : str 
        Style à appliquer, sous forme de caractères spéciaux (ex: '\033[95m') 
    titre : str 
        Titre à colorer
    texte : str
        Texte à afficher
    
    Returns
    -------
    txt : str
        Chaine avec style.

    """
    txt  = style + f'{titre}\n' + '-'*len(titre) + bcolors.ENDC
    txt += f'\n{texte}'
    return txt

def print_value_counts(df,col_list=None,exclude=None):
    """Affiche les valeurs uniques des colonnes qualitatives d'un DataFrame.
        
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe à afficher
    col_list : list, optional
        Liste des colonnes à afficher
    """
    print('Légende :\n')
    info = applique_style_paragraphe_1(bcolors.OKBLUE+bcolors.BOLD,'Colonne numérique')
    print(info)
    info = applique_style_paragraphe_1(bcolors.HEADER+bcolors.BOLD,'Colonne catégorielle')
    print(info)
    print('-'*100)
    if not col_list: 
        col_list = df.columns
    if not exclude:
        exclude = []
    for col in set(col_list)-set(exclude):
        if df[col].dtypes in ['float64']: # pas les int64, car en tant que value_counts() on peut vouloir les nb de n 
            print('\n\n'+applique_style_paragraphe_1(bcolors.OKBLUE+bcolors.BOLD,f'{col}'))
        else:
            print('\n\n'+applique_style_paragraphe_1(bcolors.HEADER+bcolors.BOLD,f'{col}',f'{df[col].value_counts()}'))
    return None


def print_nan_rates(nan_rate, seuil_50=None, seuil=None):
    """Affiche le diagramme des valeurs manquantes d'un dataframe, calculées.
    
    Les valeurs manquantes sont précédemment calculées dans `nan_rates`.

    Parameters
    ----------
    nan_rate : pandas.DataFrame
        DataFrame contenant les valeurs manquantes d'un autre dataframe.
    seuil_50 : bool, optional
        Flag d'affichage de l'axe des 50% de NaN.
    seuil : float, optional
        Valeur de seuil à afficher sous forme d'axe .
    """
    fig, ax = plt.subplots(1, 1)
    
    major_tick = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    minor_tick = [ 5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    ax.set_xticks(major_tick) # Grid
    ax.set_xticks(minor_tick, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=1)
    
    # Création du diagramme 
    nan_rate.plot(kind='barh',rot=0,grid=True,label='taux de NaN')
    # Affichage de l'axe x = 50% 
    if seuil_50:
        plt.axvline(x=50.,color='g',label='50%')
    # Affichage de l'axe x = seuil 
    if seuil:
        plt.axvline(x=seuil,color='r',label='seuil fixé')
    plt.legend(loc='lower right')
    plt.title(f'Taux de NaN par colonne');


def display_external_images(images_list, axis=False):
    """Affichage d'une liste d'images à partir de leurs chemins.
    
    Parameters
    ----------
    images_list : list
        Liste des chemins vers les images.
    axis : bool, optional
        Si True, spécifie que les axes doivent être affichés, par défaut non.
    """
    img = []
    for im in images_list:
        img.append(mpimg.imread(im))
    # display images
    n = len(img)
    if n<=1:
        plt.imshow(img[0]);
    else:
        fig, ax = plt.subplots(1,n)
        for ind in range(n):
            ax[ind].imshow(img[ind]);
    if axis:
        plt.axis('on')
    else:
        plt.axis('off')
    plt.show();


def order_df_values(df, col, seuil=0.75, max_bar=50, kind='barh'):
    """Classement des valeurs catégorielles jusqu'au seuil fixé en fréquence cumulée ou nombre fixé.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe à afficher graphiquement.
    col : str
        Colonne à afficher.
    seuil : float, optional
        Seuil maximal cumulé à afficher.
    max_bar : int, optional
        Nombre maximal de tuyaux à afficher.

    Returns
    -------
    vp : pandas.Series
        Valeurs principales.
    cles : list
        clés des premières valeurs déterminées.
    flag_seuil : bool
        Flag de seuil de pourcentage cumulé rencontré.
    flag_max_bar : bool
        Flag de nombre max de valeurs rencontré.
    kind : str, optional
        Type horizontal ou vertical du diagramme (surtout pour l'ordre croissant ou décroissant des barres).
    """
    vc = df[col].value_counts()
    sum = vc.values.sum()
    s = 0.
    n = 0
    flag_seuil   = False
    flag_max_bar = False
    cles = []
    for cle, val in zip(vc.index, vc.values):
        s += val
        n += 1
        cles.append(cle)
        if s/sum > seuil: # la nouvelle valeur a fait dépasser le seuil
            flag_seuil = True
            break
        if n>max_bar: # on a atteint le nb max de barres
            flag_max_bar = True
            break
    if kind=='barh':
        vp = vc[cles[::-1]]
    else:
        vp = vc[cles]
    return vp, cles, flag_seuil, flag_max_bar


def trace_tuyaux_principaux(df, col, seuil=0.75, max_bar=50, kind='barh', rot=0, show_nan=True, xtrot=90):
    """Tracé de diagramme en tuyaux d'orgue montrant les fréquences d'une variable catégorielle, par ordre décroissant.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe à afficher graphiquement.
    col : str
        Colonne à afficher.
    seuil : float, optional
        Seuil maximal cumulé à afficher.
    max_bar : int, optional
        Nombre maximal de tuyaux à afficher.
    kind : str, optional
        Type horizontal ou vertical du diagramme (surtout pour l'ordre croissant ou décroissant des barres).
    rot : bool, optional
        Option de rotation du diagramme.
    show_nan : bool, optional
        Option d'affichage de la valeur NaN comme valeur catgéorielle.
    xtrot : , optional
        Option de rotation des étiquettes.

    Returns
    -------
    df2 : pandas.DataFrame
        Dataframe contenant les valeurs principales.
    """
    # Classement des valeurs catégorielles jusqu'au seuil fixé en fréquence cumulée ou nombre fixé
    vp, cles, flag_seuil, flag_max_bar = order_df_values(df,col,seuil=seuil,max_bar=max_bar,kind=kind)

    # Taux de NaN
    nan_rate = df[col].isna().sum(axis=0)/df.shape[0]*100.
    
    # Tracé
    fig, ax = plt.subplots(1, 1)
    vp.plot(kind=kind,rot=rot)
    plt.grid('on')
    if flag_seuil:
        titre = f'{col} : catégories principales représentant {seuil*100.}% des valeurs'
    elif flag_max_bar:
        titre = f'{col} : catégories principales ({max_bar} premières)'
    else:
        titre = f'{col} : catégories'
    if show_nan:
        titre += f' - NaN : {nan_rate:.2f}%'
    plt.title(titre)
    if vp.index.dtype in ['object']:
        xticks = [val if len(val)<=50 else val[:50]+'..' for val in vp.index]
        if kind=='barh':
            ax.set_yticklabels(xticks)
        else:
            ax.set_xticklabels(xticks)
    plt.xticks(rotation=xtrot,ha='right')
    df2 = df.copy()
    df2 = df2.loc[df2[col].isin(cles)]
    return df2


def plot_distrib(df, num_cols, nbins=50, facecolor='orange', exclude=None):
    """Trace une distribution.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe à afficher graphiquement.
    num_cols : list
        Colonnes numériques à afficher.
    nbins : int, optional
        Nombre de barres de l'histogramme.
    facecolor : str; optional
        Couleur des barres.
    exclude : list
        Liste des colonnes à exclure de l'affichage.
    """
    if not exclude:
        exclude = []
    for var in set(num_cols)-set(exclude):
        vals = df[var].values
        n, bins, patches = plt.hist(vals, nbins, density=False, facecolor=facecolor, alpha=0.75) #'c'
        plt.xlabel(f'{var}')
        plt.ylabel('n')
        plt.title(f'Histogramme de {var}')
        plt.grid(True)
        plt.show()


def compare_2df_distrib_num(df1, df2, num_cols, label1='Inliers', label2='Outliers', colors=None):
    """Compare les distributions de deux dataframes, sous forme de boxplots.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        Premier dataframe à comparer.
    df2 : pandas.DataFrame
        Second dataframe à comparer.
    num_cols : list
        Colonnes numériques à considérer.
    label1 : str, optional
        Label à afficher en légende, pour les inliers par exemple.
    labelE : str, optional
        Label à afficher en légende, pour les outliers par exemple.
    colors : list or None
        Liste des couleurs à utiliser pour les boxplots.
    """
    if not colors:
        colors = ['mediumslateblue','red']
    n = len(num_cols)
    plt.figure(figsize=(16,8*n))
    for ind in range(n):
        cat = num_cols[ind]

        plt.subplot(n+1,2,2*ind+1)
        df1[cat].hist(density=True,color=colors[0],bins=100,alpha=0.8,label=label1)
        df2[cat].hist(density=True,color=colors[1],bins=100,alpha=0.8,label=label2)
        plt.legend()
        plt.title(f'{label1} ({df1.shape[0]})/{label2} ({df2.shape[0]} samples) - '+cat);

        plt.subplot(n+1,2,2*ind+2)
        df1b = df1.copy()
        df1b['DF'] = label1
        df2b = df2.copy()
        df2b['DF'] = label2
        df = pd.concat([df1b,df2b],axis=0)
        print(f"df : {df}")
        if df.empty:
            print(bcolors.BOLD+bcolors.HEADER+"Aucun outlier !"+bcolors.ENDC)
        else:
            sns.boxplot(data=df, x=cat, y='DF', palette={label1: "mediumslateblue", label2: "red"}, 
                        showmeans=True, showfliers=False) # 'gist_rainbow_r', width=0.8, notch=True, palette='coolwarm'
            plt.title(f'{label1} ({df1.shape[0]})/{label2} ({df2.shape[0]} samples) - '+cat);
            plt.grid('on')    


def compare_2df_repartition_categ(df1, df2, categ_cols, label1='Inliers', label2='Outliers', colormap='tab20c'):
    """Compare les répartitions catégorielles de deux dataframes, sous forme de pieplots.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        Premier dataframe à comparer.
    df2 : pandas.DataFrame
        Second dataframe à comparer.
    categ_cols : list
        Colonnes catégorielles à considérer.
    label1 : str, optional
        Label à afficher en légende, pour les inliers par exemple.
    labelE : str, optional
        Label à afficher en légende, pour les outliers par exemple.
    colormap : str
        Colormap à utiliser pour les pieplots.
    """
    n = len(categ_cols)
    plt.figure(figsize=(12,9*n))
    for ind in range(n):
        cat = categ_cols[ind]
        plt.subplot(n+1,2,2*ind+1)
        df1[cat].value_counts().plot.pie(colormap=colormap) # ,autopct=lambda p: '{:.0f}'.format(p)
        plt.title(label1+' - '+cat+f" - {df1.shape[0]} samples")
        plt.subplot(n+1,2,2*ind+2)
        df2[cat].value_counts().plot.pie(colormap=colormap) # ,autopct=lambda p: '{:.0f}'.format(p)
        plt.title(label2+' - '+cat+f" - {df2.shape[0]} samples");


def compare_in_out_liers(df, lier_var, lier_cat, lier_cat_val, categ_cols, num_cols, side='high', seuil=None):
    """Compare les distributions numériques et répartions catégorielles de toutes les variables dont l'une fait l'objet
    d'inliers et outliers.
    
    On commence par déterminer les inliers et outliers. Puis on trace les distributions et répartitions des autres variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe à étudier.
    categ_cols : list
        Colonnes catégorielles à considérer.
    lier_var : str
        Variable faisant l'objet d'inliers et outliers.
    lier_cat : str
        Catégorie pour laquelle une valeur catégorielle rend une distribution numérique avec des outliers. 
    lier_cat_val : str
        Valeur catégorielle pour laquelle une distribution numérique a des outliers. 
    categ_cols : list
        Liste des colonnes catégorielles
    num_cols : list
        Liste des colonnes numériques
    side : str, optional
        Côté à considérer (outlier par valeur inférieure ou par valeur supérieure = par défaut)
    seuil : float or None
        Seuil de considération d'outlier TODO : ça va pas marcher si on rentre une seule valeur
    """
    if not seuil:
        Q1,Q3 = np.percentile(df[lier_var], [25., 75.])
        IQR = Q3 - Q1
        seuil_min = Q1 - 1.5*IQR
        seuil_max = Q3 + 1.5*IQR
        # print(seuil_min)
        print(f"Seuils in/out liers :[{seuil_min:.4f} ; {seuil_max:.4f}]")
    else:
        seuil_min = seuil
        seuil_max = seuil

    df_inliers = df.loc[(df[lier_cat]==lier_cat_val) & (df[lier_var]>=seuil_min) & (df[lier_var]<=seuil_max)]
    if side=='low':
        df_outliers = df.loc[(df[lier_cat]==lier_cat_val) & (df[lier_var]<seuil_min)]        
    else:
        df_outliers = df.loc[(df[lier_cat]==lier_cat_val) & (df[lier_var]>seuil_max)]

    # Statistiques basiques
    print(f'df_inliers :\n------------\n{df_inliers.describe()}\n')
    print(f'df_outliers :\n-------------\n{df_outliers.describe()}\n')

    # Visualisations des variables catégorielles
    compare_2df_repartition_categ(df_inliers,df_outliers,categ_cols)

    # Visualisations des variables numériques
    compare_2df_distrib_num(df_inliers,df_outliers,num_cols)

    return df_inliers, df_outliers