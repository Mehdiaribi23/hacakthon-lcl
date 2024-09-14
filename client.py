
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Spécifiez le chemin du fichier CSV
file_path = 'clients_avec_offres (15).csv'

# Lire le fichier CSV dans un DataFrame
df = pd.read_csv(file_path)

# Afficher les premières lignes du DataFrame pour vérifier
print(df.head())


# Visualiser les premières lignes
df.head()

# Voir les informations des colonnes
df.info()


# Vérifier les valeurs manquantes
df.isnull().sum()

# Par exemple, supprimer les lignes avec des valeurs manquantes
df = df.dropna()




# Supposons que 'Offre attribuée' est la variable cible
X = df.drop(['Offre attribuée'], axis=1)  # Variables explicatives
y = df['Offre attribuée']  # Cible

# Convertir les données catégoriques si nécessaire
X = pd.get_dummies(X)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Créer le modèle
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
clf.fit(X_train, y_train)

# Faire des prédictions
y_pred = clf.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle: {accuracy * 100:.2f}%")



# Rapport de classification
print(classification_report(y_test, y_pred))



from sklearn.model_selection import GridSearchCV

# Définir une grille d'hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Meilleur modèle
print(grid_search.best_params_)



import joblib

# Sauvegarder le modèle entraîné
joblib.dump(clf, 'modele_random_forest.pkl')





# Charger le modèle
model = joblib.load('modele_random_forest.pkl')

# Faire de nouvelles prédictions
y_new_pred = model.predict(X_test)

print (y_new_pred)



def predire_offre(client_id):
    # Filtrer les caractéristiques du client
    client_data = df[df['ID client'] == client_id].drop(['Offre attribuée', 'ID client'], axis=1)

    # Convertir les variables catégoriques en numériques
    client_data = pd.get_dummies(client_data)

    # Assurez-vous que les colonnes du client correspondent au jeu d'entraînement
    # Utilisez X (le DataFrame avant la normalisation) pour reindexer les colonnes
    client_data = client_data.reindex(columns=X.columns, fill_value=0)

    # Appliquer la normalisation après avoir aligné les colonnes
    client_data = scaler.transform(client_data)

    # Faire la prédiction
    predicted_offer = clf.predict(client_data)

    # Retourner le résultat
    return predicted_offer[0]

# Appel de la fonction pour un ID client spécifique
offre_predite = predire_offre('00-07-F3-D5')
print(f"L'offre prédite est : {offre_predite}")


from sklearn.metrics import accuracy_score, classification_report

# Prédictions sur l'ensemble de test
y_test_pred = clf.predict(X_test)

# Calculer la précision
precision = accuracy_score(y_test, y_test_pred)
print(f"La précision du modèle est : {precision * 100:.2f}%")

# Afficher un rapport de classification détaillé
print("Rapport de classification :")
print(classification_report(y_test, y_test_pred))


# Avant de normaliser les données, conserver les IDs des clients dans l'ensemble de test
X_train_original, X_test_original, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Conserver l'index des clients avant la normalisation
client_ids_test = df.loc[X_test_original.index, 'ID client']

# Appliquer la normalisation après avoir divisé les données
X_train = scaler.fit_transform(X_train_original)
X_test = scaler.transform(X_test_original)

# Faire les prédictions sur l'ensemble de test
y_test_pred = clf.predict(X_test)

# Créer un DataFrame avec les IDs des clients, les vraies valeurs et les prédictions
resultats = pd.DataFrame({
    'ID client': client_ids_test,
    'Offre réelle': y_test,
    'Offre prédite': y_test_pred
})

# Afficher les résultats
print(resultats)

# Afficher la précision du modèle
precision = accuracy_score(y_test, y_test_pred)
print(f"La précision du modèle est : {precision * 100:.2f}%")


import matplotlib.pyplot as plt


# Compter les valeurs réelles et prédites
real_offer_counts = resultats['Offre réelle'].value_counts()
predicted_offer_counts = resultats['Offre prédite'].value_counts()

# Création du diagramme à barres comparatif
labels = real_offer_counts.index  # Les types d'offres
x = range(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))

# Barres pour les offres réelles
ax.bar(x, real_offer_counts, width=0.4, label='Offres réelles', align='center')

# Barres pour les offres prédites (décalées pour comparaison)
ax.bar([i + 0.4 for i in x], predicted_offer_counts, width=0.4, label='Offres prédites', align='center')

# Ajouter les labels
ax.set_xticks([i + 0.2 for i in x])
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel('Nombre d\'offres')
ax.set_title('Comparaison des offres réelles et prédites')

# Ajouter une légende
ax.legend()

# Afficher le diagramme
plt.tight_layout()
plt.show()



