# -*- coding: utf-8 -*-
"""
Created on 2022/06/07

@author: Cyril G.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from DisplotData import bcolors
import Funktiuns


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



def display_cv_results(problem_kind, clf, score, display_results_text=0,
                       algo_surname=None):
    """Affiche les résultats d'une validation croisée.
    
    Parameters
    ----------
    problem_kind : str
        Type de problème traité (régression/classification/...).
    clf : sklearn.model_selection.GridSearchCV
        Modèle de validation croisée.
    score : float
        Score à afficher.
    display_results_text : int, optional
        Flag d'affichage des résultats textuels.
    algo_surname : str
        Surnom de l'algorithme traité.
    """
    if display_results_text>0:
        # Afficher le(s) hyperparamètre(s) optimaux
        if algo_surname=='NeuralNetwork' : # TODO
            s = 0. # TODO
            b = {}
        else:
            s = clf.cv_results_['mean_test_score'].max()
            b = clf.best_params_
        if problem_kind=='regression': s = -s # TODO allez hop

        print(bcolors.BOLD+f"Best hyperparameters on the train set: {b}. " + \
              f"Score (val) : {s:.3e}"+bcolors.ENDC)

    if display_results_text>1 and algo_surname!='NeuralNetwork':
        # Afficher les performances correspondantes
        print("Résultats de la validation croisée :")
        for mean, std, params in zip(
                clf.cv_results_['mean_test_score'], # score moyen
                clf.cv_results_['std_test_score'],  # écart-type du score
                clf.cv_results_['params']           # valeur de l'hyperparamètre
            ):

            print("{} = {:.3e} (+/-{:.3e}) for {}".format(score,mean,std*2,params)
            )


def display_cv_graphs(problem_kind,mdl):
    """Affiche les résultats graphiques d'une validation croisée.
    
    Parameters
    ----------
    problem_kind : str
        Type de problème traité (régression/classification/...).
    mdl : sklearn...
    """
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    if problem_kind=='classification':
        pass
    else: # regression
        pass


# ------------------------------------------------------------------------------
# Fonctions OpenClassRooms (Nicolas Rangeon), merci à eux
# ------------------------------------------------------------------------------
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None, label_size=None,
                    title_prefix='', title_suffix=''):
    """Affiche les cercles de corrélation d'une PCA.
    
    Parameters
    ----------
    pcs : numpy.array
        Coordonnées des composantes principales
    n_comp : int
        Nombre de composantes principales de la PCA
    pca : sklearn.decomposition.PCA
        Modèle de PCA.
    axis_ranks : list
        Liste des plans factorils à afficher (ex : [(0,1)], [(0,1), (2,3)] etc.).
    labels : list, optional
        Liste des noms de variables.
    label_rotation : float, optional
        Angle de rotation des labels des noms de variables.
    lims : list, optional
        Limites xmin, xmax, ymin, ymax=.
    label_size : int, optional
        Taille du label des noms de variables.
    title_prefix : , optional
        Préfixe du titre du graphique.
    title_suffix : , optional
        Suffixe du titre du graphique.
    """
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
            if not label_size:
                label_size=10
            # initialisation de la figure
            fig, ax = plt.subplots() #figsize=(10,10)

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            xmin, xmax, ymin, ymax = -1, 1, -1, 1 # DEBUG
#             xmin, xmax, ymin, ymax = -.3, 0.3, -0.3, 0.3
            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize=label_size, ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title(title_prefix+"Correlation circle (F{}, F{})".format(d1+1, d2+1)+title_suffix)
            plt.show(block=False)


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None,
                            marker='o', markerscale=None,
                            s=72, xlim=None, ylim=None,
                            title_prefix='', title_suffix=''):
    """Affiche la projection des individus dans les plans principaux d'une PCA.
    
    Parameters
    ----------
    X_projected : numpy.array
        Coordonnées des composantes points projetés.
    n_comp : int
        Nombre de composantes principales de la PCA
    pca : sklearn.decomposition.PCA
        Modèle de PCA.
    axis_ranks : list
        Liste des plans factorils à afficher (ex : [(0,1)], [(0,1), (2,3)] etc.).
    labels : list, optional
        Liste des noms de variables.
    alpha : float, optional
        Coefficient alpha du tracé graphique.
    illustrative_var : list, optional
        Liste des labels d'une variable illustrative.
    marker : str, optional
        Marqueur pour les points.
    s : int, optional
        Taille du label des noms de variables.
    xlims : list, optional
        Limites xmin, xmax.
    ylims
        Limites ymin, ymax.
    title_prefix : , optional
        Préfixe du titre du graphique.
    title_suffix : , optional
        Suffixe du titre du graphique.
    """
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure() #figsize=(15,15)
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha, marker=marker, s=s)
            else:
                illustrative_var = np.array(illustrative_var)
                # print(np.unique(illustrative_var))
                for value in np.unique(illustrative_var):
                    # print(value)
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value, 
                                marker=marker,s=s)

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='10', ha='center', va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            if xlim:
                plt.xlim(xlim)
            else:
                plt.xlim([-boundary,boundary])
            if ylim:
                plt.ylim(ylim)
            else:
                plt.ylim([-boundary,boundary])

        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title(title_prefix+"Sample projection on (F{}, F{})".format(d1+1, d2+1)+title_suffix)
            if not illustrative_var is None:
                if markerscale is not None:
                    plt.legend(prop={'size': 15}, markerscale=markerscale)
                else:
                    plt.legend(prop={'size': 15})
            
            plt.show(block=False)

def display_scree_plot(pca, title_prefix='', title_suffix=''):
    """Affiche l'éboulis des valeurs propres d'une PCA."""
    scree = pca.explained_variance_ratio_*100
    fig = plt.figure() 
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Component")
    plt.ylabel("Inertia percentage")
    plt.title(title_prefix+"Eigenvalue scree plot"+title_suffix)
    plt.show(block=False)


def plot_dendrogram(Z, names):
    """Affiche le dendrogramme d'un clustering hiérarchique."""
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance') 
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()


def display_predictions_vs_truth(X_train, y_train, y_train_pred, # X_train possiblement projeté sur sa PCA !
                                 X_test, y_test, y_test_pred,    # X_test possiblement projeté sur la PCA de X_train !
                                 feat_cols, target_col,
                                 xlim_train=None, ylim_train=None,
                                 xlim_test=None, ylim_test=None,
                                 worst_pred_indexes=None,
                                 algo_name=""):
    """Affiche le graph ground/truth d'une régression, ainsi que les prédictions en fonction de l'axe principal d'une PCA.
    
    Parameters
    ----------
    X_train : np.array
        Matrice du dataset d'entraînement. 
    y_train : np.array
        Target du dataset d'entraînement. 
    y_train_pred : np.array
        Target prédite sur le dataset d'entraînement. 
    X_test : np.array
        Matrice du dataset de test. 
    y_test : np.array
        Target du dataset de test. 
    y_test_pred : np.array
        Target prédite sur le dataset de test. 
    feat_cols : list
        Liste des colonnes formant les features.
    target_col : str
        Nom de colonne de la target.
    xlim_train : list, optional
        Limites xmin, xmax pour le graphe du train set.
    ylim_train : list, optional
        Limites ymin, ymax pour le graphe du train set.
    xlim_test : list, optional
        Limites xmin, xmax pour le graphe du test set.
    ylim_test : list, optional
        Limites ymin, ymax pour le graphe du test set.
    worst_pred_indexes : list, optional
        Liste des labels des plus mauvaises prédictions.
    algo_name : str, optional
        Nom de l'algorithme utilisé.
    """
    target_col_pred = target_col+'_pred'

    rows = 2
    cols = 2
    plt.rcParams['figure.figsize'] = [8*rows, 6*cols]
    fig = plt.figure()

    # Calcul d'erreurs
    RMSE_train = math.sqrt(mean_squared_error(y_train,y_train_pred))
    MAE_train  = mean_absolute_error(y_train,y_train_pred)
    R2_train   = r2_score(y_train,y_train_pred)
    RMSE_test  = math.sqrt(mean_squared_error(y_test,y_test_pred))
    MAE_test   = mean_absolute_error(y_test,y_test_pred)
    R2_test    = r2_score(y_test,y_test_pred)

    # Erreurs relatives
    zero_train = np.zeros(y_train.shape)
    zero_test  = np.zeros(y_test.shape)
    n_train = y_train.shape[0]
    n_test  = y_test.shape[0]
    RRMSE_train = RMSE_train / math.sqrt(mean_squared_error(y_train,zero_train)*n_train)
    RMAE_train  = MAE_train / (mean_absolute_error(y_train,zero_train)*n_train)
    RRMSE_test  = RMSE_test / math.sqrt(mean_squared_error(y_test,zero_test)*n_test)
    RMAE_test   = MAE_test / (mean_absolute_error(y_test,zero_test)*n_test)

    # Titre supérieur
    feat_str = Funktiuns.format_long_string(str(feat_cols), 80, ',', warning=True, elude=True)
    suptitle = f"Model {algo_name} with\n{feat_str}\n↦ {target_col}"
    fig.suptitle(suptitle, fontsize=16)
    
    grid = plt.GridSpec(rows, cols, wspace=0.15, hspace=0.3)

    fig_ax1 = fig.add_subplot(grid[0, 0])
    fig_ax2 = fig.add_subplot(grid[0, 1])
    # fig_ax2.set_axis_off()
    fig_ax3 = fig.add_subplot(grid[1, 0])
    fig_ax4 = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(top=0.8)

    # Train set
    fig_ax1.plot(y_train,y_train,c='blue',label="Ground truth")
    fig_ax1.scatter(y_train,y_train_pred,marker='+',c='orange',label="Predictions")
    if ylim_train:
        fig_ax1.set_xlim(ylim_train)
        fig_ax1.set_ylim(ylim_train)
    fig_ax1.legend()
    fig_ax1.grid()
    errs_train = f"Rel. RMSE: {RRMSE_train:.2e} | Rel. MAE: {RMAE_train:.2e} | $R²$: {R2_train:.2f}"
    fig_ax1.set_title(f"Train set - predictions vs truth\n"+errs_train)
    fig_ax1.set_xlabel('y_true')
    fig_ax1.set_ylabel('y_pred')

    # Test set
    fig_ax2.plot(y_test,y_test,c='blue',label="Ground truth")
    fig_ax2.scatter(y_test,y_test_pred,marker='+',c='orange',label="Predictions")
    if ylim_test:
        fig_ax2.set_xlim(ylim_test)
        fig_ax2.set_ylim(ylim_test)
    fig_ax2.legend()
    fig_ax2.grid()
    errs_test = f"Rel. RMSE: {RRMSE_test:.2e} | Rel. MAE: {RMAE_test:.2e} | $R²$: {R2_test:.2f}"
    fig_ax2.set_title(f"Test set - predictions vs truth\n"+errs_test)
    # Affichage des index des pires prédictions
    # b = (y_test_pred[target_col_pred].max()-y_test_pred[target_col_pred].min())/20.
    b = 0
    if worst_pred_indexes:
        for ind in worst_pred_indexes:
            fig_ax2.annotate(str(ind), (y_test.loc[ind,target_col]-5*b, y_test_pred.loc[ind,target_col_pred]+b),fontsize=8)
    fig_ax2.set_xlabel('y_true')
    fig_ax2.set_ylabel('y_pred')

    # Train set
    fig_ax3.scatter(X_train, y_train, marker='x', c='green', label="y_true",alpha=0.6,s=14)
    fig_ax3.scatter(X_train, y_train_pred, marker='+', c='red', label="y_pred",alpha=0.6)
    if xlim_train:
        fig_ax3.set_xlim(xlim_train)
        fig_ax3.set_ylim(ylim_train)
    fig_ax3.legend()
    fig_ax3.grid()
    fig_ax3.set_title(f"Train set - predictions vs truth\n"+errs_train)
    fig_ax3.set_xlabel('X1')
    fig_ax3.set_ylabel('y')

    # Test set
    fig_ax4.scatter(X_test,y_test,marker='x',c='green',label="y_true",alpha=0.6,s=14) # ,s=10
    fig_ax4.scatter(X_test,y_test_pred,marker='+',c='red',label="y_pred",alpha=0.6) # ,s=10
    if xlim_test:
        fig_ax4.set_xlim(xlim_test)
        fig_ax4.set_ylim(ylim_test)
    fig_ax4.legend()
    fig_ax4.grid()
    fig_ax4.set_title(f"Test set - predictions vs truth\n"+errs_test)
    # Affichage des index des pires prédictions
    # a = (X_test[0].max()-X_test[0].min())/20.
    # b = (y_test_pred[target_col_pred].max()-y_test_pred[target_col_pred].min())/20.
    a = 0
    b = 0
    if worst_pred_indexes:
        for ind in worst_pred_indexes:
            fig_ax4.annotate(str(ind), (X_test.loc[ind,0]-a, y_test_pred.loc[ind,target_col_pred]+b),fontsize=8)
    fig_ax4.set_xlabel('X1')
    fig_ax4.set_ylabel('y')

    plt.show()
