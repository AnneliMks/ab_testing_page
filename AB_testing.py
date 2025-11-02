#importation des bibliothèques
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats import proportion
from statsmodels.stats.power import NormalIndPower
np.set_printoptions(legacy='1.25')


#Exploration des données
df = pd.read_csv("./ab_data.csv.zip", sep = ",",
                 parse_dates=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
df.head()

df.describe()
#En analysant les résultats on voit déjà, une moyenne assez basse (0.11) pour "converted" ce qui
# démontre que la part de 0 et plus élevée que la part de 1
df.info()
df.isna().sum() #0 valeurs manquantes dans toutes les colonnes

#Transformation des colonnes

df = df.astype({"group": "category", "landing_page": "category", "converted": "category"})

# Hypothèses : on veut savoir si taux de conversion est différents en fonction des versions
#Donc H0 : p_o = p_n (taux de conversion pareil qq soit la page "old" ou "new")
# H1: p_o != p_n on choisit alpha = 5%

df = df.drop_duplicates(subset = "user_id", keep = "first")

print("Nombre de personnes dans chaque groupe :", df["group"].value_counts())

#On observe que le nombre de personnes par groupe est parfaitement équilibré : 145232.

#On commence quand même par faire des analyses graphique

plt.figure(figsize=(10,5))
sns.countplot(df, x = "group",hue="converted",stat= "count" )
#plt.show()
# On veut qu'en général le nombre de personnes "converted" est très faible dans les deux groupes
# mais on voit également dans les 2 groupe c'est assez équilibré => il faut faire des tests statistiques
#car la visualisation ne suffit pa

#Taux de conversion pour l'ancienne page
df_old_page = df[df["landing_page"]=="old_page"]

taux_old_page = round(len(df_old_page[df_old_page["converted"]== 1])/len(df_old_page),4)
print("Le taux de conversion pour l'ancienne page est de :",
      taux_old_page*100,"%")

df_new_page = df[df["landing_page"]=="new_page"]

taux_new_page = round(len(df_new_page[df_new_page["converted"]== 1])/len(df_new_page),4)
print("Le taux de conversion pour la nouvelle  page est de :",
      taux_new_page*100,"%")

#Donc le taux de convertion pour l'ancienne page était d'environ 12%/
#Il faut une hypothèse solide : par exemple la nouvelle page va augmenter de au moins 5% le taux de conversion

#On s'assure qu'il y a suffisamment de données pour chaque groupe afin d'avoir un résultat stable
# même si a priori cela semble plutôt bon.

alpha = 0.05
size_effet = proportion.proportion_effectsize(taux_old_page, taux_old_page+0.05)
taille_min = round(NormalIndPower().solve_power(size_effet, power = 0.9, alpha = alpha))
#On veut donc savoir qu'elle taille minimum dans chaque groupe pour avoir 90% (puissance) de chance
# de détecter une différence avec une erreur de 5%.

print(f"La taille de l'échantillon doit être au minimum de :{taille_min}. "
      f"Or {taille_min} < {df['group'].value_counts().iloc[0]}")

Nb_old_page = len(df_old_page)
Nb_new_page = len(df_new_page)

Nb_converted_new_page = len(df_new_page[df_new_page["converted"]== 1])
Nb_converted_old_page = len(df_old_page[df_old_page["converted"]== 1])


#On calcule maintenant le z-test

stat, pvalue = proportion.proportions_ztest([Nb_converted_old_page, Nb_converted_new_page],
                                             [ Nb_old_page, Nb_new_page ])

print(f'Statistic: {stat}, p-value: {pvalue}')

if pvalue < alpha :
    print("Il ya une différence significative dans la taux de converstion entre les deux page")
else:
    print("Il n'y a pas de différences significative dans le taux de converstion entre les deux pages")

ci_low, ci_upp = proportion.proportion_confint([Nb_converted_old_page, Nb_converted_new_page],
                                             [ Nb_old_page, Nb_new_page ],
                                               alpha=alpha)

print(f"IC pour groupe de contrôle (old page) : [{round(ci_low[0],4)*100}, {round(ci_upp[0],4)*100}]")
print(f"IC pour groupe de traitement (new page) : [{round(ci_low[1],4)*100}, {round(ci_upp[1],4)*100}]")

#On arrive à plusieurs conclusions :
#1. En observant l'échantillon, on voyait dejà que le taux de convertion était meilleur pour l'ancienne page
#2. Par l'inférence statistique, on a montré qu'il n'y avait pas de différences significatives entre les 2 groupes
#3. L'objectif cible pour la nouvelle page n'étant pas atteinte (pas dans l'IC), il vaut mieux garder l'ancienne page