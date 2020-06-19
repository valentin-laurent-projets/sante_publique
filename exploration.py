import pandas as pd
import numpy as np
from collections import Counter



def premier_quartile(data_frame,colonne):
    """Pour une variables quantitative. Retourne la valeur du premier quartile. Colonne est le nom de la colonne."""
    return data_frame[colonne].quantile(q=0.25)




def troisieme_quartile(data_frame,colonne):
    """Pour une variables quantitative. Retourne la valeur du troisième quartile. Colonne est le nom de la colonne"""
    return data_frame[colonne].quantile(q=0.75)




def inter_quartile(data_frame,colonne):
    """Retourne l'écart inter-quartile."""
    return troisieme_quartile(data_frame,colonne)-premier_quartile(data_frame,colonne)




def outliers_inf(data_frame,colonne,borne_inf=True,nouvelle_borne_inf=0):
    """Pour une hauteur en mètre par exemple, un outliers négatif n'a aucun sens.
    
    C'est pourquoi on peut choisir sa borne inférieure en mettant borne_inf=False et en choisissant une nouvelle_borne_inf.
    Utilité pour les fonctions suivantes qui se servent de cette fonction."""
    if borne_inf:
        return premier_quartile(data_frame,colonne)-1.5*inter_quartile(data_frame,colonne)
    else:
        return nouvelle_borne_inf

    
    
def outliers_sup(data_frame,colonne,borne_sup=True,nouvelle_borne_sup=0):
    """Pour des cas particuliers, on peut choisir la borne supérieure en mettant borne_sup=False et en choisissant une nouvelle_borne_sup.
    
    Utilité pour les fonctions suivantes qui se servent de cette fonction."""
    if borne_sup:
        return troisieme_quartile(data_frame,colonne)+1.5*inter_quartile(data_frame,colonne)
    else:
        return nouvelle_borne_sup

    
    
    
def sans_outliers(data_frame,colonne,borne_inf=True,nouvelle_borne_inf=0,borne_sup=True,nouvelle_borne_sup=0):
    """Retourne la colonne sélectionnée sans les valeurs outliers. """
    
    mask=(data_frame[colonne]>=outliers_inf(data_frame,colonne,borne_inf,nouvelle_borne_inf))&(data_frame[colonne]<=outliers_sup(data_frame,colonne,borne_sup,nouvelle_borne_sup))
    
    return data_frame.loc[mask,colonne]




def value_counts_hist(data_frame,colonne,bins,borne_inf=True,nouvelle_borne_inf=0,borne_sup=True,nouvelle_borne_sup=0):
    """
    Cette fonction permet d'afficher l'histogramme d'une variable QUANTITATIVE CONTINUE DISCRETISEE en n=bins intervalles.
    
    Renvoie la même chose que la fonction value_counts() mais sous forme de DataFrame(counts_df) dans lequel les effectifs 
    ne sont pas triés du plus grand au plus petit ET où les intervalles sont des strings."""
    data_frame_disc=pd.cut(sans_outliers(data_frame,colonne,borne_inf,nouvelle_borne_inf,borne_sup,nouvelle_borne_sup), bins=bins)
    counts_df=pd.DataFrame()
    counts_df["intervalles"]=data_frame_disc.value_counts(sort=False).index
    counts_df["effectifs"]=data_frame_disc.value_counts(sort=False).values
    counts_df["intervalles"]=counts_df["intervalles"].astype('str')
    return counts_df


def most_common_words(label,nombre_de_mots=10):
    """
    Cette fonction renvoie une liste de dimension (10x2) des 10 mots (par défaut) les plus présents parmi les 
    modalités d'une variable QUALITATIVE et leur occurence.
    
    Utilité de cette fonction natamment pour les variables qualitatives qui prennent beaucoup de modalités."""
    words=[]
    for lab in label:
        lab=str(lab)
        words+=lab.split(" ")
    counter=Counter(words)
    return counter.most_common(nombre_de_mots)




def detect_words(values,label,nombre_de_mots=10):
    """
    values=data["exemple"] avec "exemple" une variable qualitative ayant beaucoup de modalites.
    label=data["exemple"].values
    
    Cette fonction renvoie "result" qui est l'équivalent de data["exemple"] mais uniquement avec les 10 (par défaut) modalités qui
    apparaissent le plus. Les autres modalités ont alors la valeur "AUTRE".
    """
    liste=most_common_words(label,nombre_de_mots)
    dictionary=dict(liste)
    result = []
    for lib in values:
        type_values = "AUTRE"
        for word, val in dictionary.items():
            if word==lib:
                type_values = word
        result.append(type_values)
    return result




def selection_variables(data, liste_substring, autres=False, autres_variables=[]):
    """
    Cette fonction permet de sélectionner des variables (d'un dataset data) qui contiendraient des
    chaines de caractères de liste_substring dans leur nom.
    
    Si autres est True alors il est possible de rajouter manuellement des variables à sélectionner 
    grâce à autres_variables.
    
    Renvoie le dataset uniquement avec les variables sélectionnées.
    """
    colonnes=data.columns.values
    variables_selectionnees=[]
    for sub in liste_substring:
        for col in colonnes:
            if (sub in col)&(col not in variables_selectionnees):
                variables_selectionnees.append(col)
    if autres:
        for autre in autres_variables:
            variables_selectionnees.append(autre)
    return data[variables_selectionnees]




def suppression_variables(data, liste_substring, keep_substring=[],autres=False, autres_variables=[], exceptions=False, exceptions_variables=[]):
    """
    Cette fonction permet de supprimer des variables (d'un dataset data) qui contiendraient des
    chaines de caractères de liste_substring dans leur nom.
    
    keep_substring permet de garder les colonnes qui contiendraient une partie d'un substring.
    Exemple: remove "_fr" keep "_from"
    
    Si exceptions est True, il est possible de préciser de variables qui ne doivent pas être
    supprimées.
    
    Si autres est True alors il est possible de rajouter manuellement des variables à supprimer 
    grâce à autres_variables.
    
    Renvoie le dataset uniquement avec les variables sélectionnées.
    """
    colonnes=data.columns.values
    variables_supprimees=[]
    for sub in liste_substring:
        if exceptions:
            for col in colonnes:
                if (sub in col)&(sub not in keep_substring)&(col not in variables_supprimees)&(col not in exceptions_variables):
                    variables_supprimees.append(col)
        else:
            for col in colonnes:
                if (sub in col)&(sub not in keep_substring)&(col not in variables_supprimees):
                    variables_supprimees.append(col)
    
    if autres:
        for autre in autres_variables:
            variables_supprimees.append(autre)
    return data.drop(variables_supprimees, axis=1)





def missing_frame(data, get_row=False, row_name='la_colonne'):
    """
    Cette fonction renvoie le tableau du nombre de valeurs manquantes par colonne.
    Il indique également le facteur de remplissage en pourcentage qui correspond au pourcentage
    de valeurs indiquées dans la colonne: 0% = aucune valeur n'est indiquée.
    """
    missing_df=pd.DataFrame(data.isna().sum().sort_values(ascending=[False]).index,columns=['nom_colonne'])
    missing_df['nbr_val_manquantes']=data.isna().sum().sort_values(ascending=[False]).values
    missing_df['facteur_remplissage']=(1-(missing_df['nbr_val_manquantes']/len(data)))*100
    if get_row==True:
        return missing_df[missing_df['nom_colonne']==row_name]
    return missing_df
    


    

def tableau_apercu(liste_tableau, liste_tableau_str):
    """
    liste_tableau=[data_exemple1,data_exemple2] 
    
    Cette fonction renvoie le nombre de lignes, de colonnes, de variables qualitatives 
    de variables quantitatives pour chaque tableau indiqué dans liste_tableau.
    """
    quick_view_df = pd.DataFrame(liste_tableau_str, columns=['nom_tableau'])
    quick_view_df['lignes'] = pd.Series(dtype=np.float64)
    quick_view_df['variables'] = pd.Series(dtype=np.float64)
    quick_view_df['nbr_var_quanti'] = pd.Series(dtype=np.float64)
    quick_view_df['nbr_var_quali'] = pd.Series(dtype=np.float64)
    for i in range(len(liste_tableau)):
        quick_view_df.loc[i,'lignes']=liste_tableau[i].shape[0]
        quick_view_df.loc[i,'variables']=liste_tableau[i].shape[1]
        quick_view_df.loc[i,'nbr_var_quanti']=len(liste_tableau[i].columns.values[liste_tableau[i].dtypes.values=="float64"])
        quick_view_df.loc[i,'nbr_var_quali']=len(liste_tableau[i].columns.values[liste_tableau[i].dtypes.values=="object"])
    return quick_view_df
    
    
    

    
    
    
    
    