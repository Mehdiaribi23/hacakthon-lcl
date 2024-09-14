import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# Fonction pour charger et prétraiter les données
def charger_et_preparer_donnees(file_path):
    # Lire le fichier CSV
    df = pd.read_csv(file_path)
    
    # Supprimer les lignes avec des valeurs manquantes
    df = df.dropna()
    
    # Séparer les variables explicatives (X) et la cible (y)
    X = df.drop(['Offre attribuée'], axis=1)
    y = df['Offre attribuée']
    
    # Convertir les données catégoriques en variables numériques
    X = pd.get_dummies(X)
    
    return X, y, df

# Fonction pour diviser les données et les normaliser
def diviser_normaliser_donnees(X, y, test_size=0.3, random_state=42):
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Conserver les index avant la normalisation
    X_test_index = X_test.index
    
    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, X_test_index

# Fonction pour entraîner le modèle
def entrainer_modele(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf

# Fonction pour optimiser les hyperparamètres du modèle
def optimiser_modele(clf, X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
    }
    
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

# Fonction pour évaluer le modèle
def evaluer_modele(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle: {precision * 100:.2f}%")
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    
    return y_pred

# Fonction pour sauvegarder le modèle
def sauvegarder_modele(clf, scaler, filename_model='modele_random_forest.pkl', filename_scaler='scaler.pkl'):
    joblib.dump(clf, filename_model)
    joblib.dump(scaler, filename_scaler)

# Fonction pour charger le modèle
def charger_modele(filename_model='modele_random_forest.pkl', filename_scaler='scaler.pkl'):
    model = joblib.load(filename_model)
    scaler = joblib.load(filename_scaler)
    
    return model, scaler

# Fonction de prédiction pour un client spécifique
def predire_offre(client_id, df, clf, scaler, X_columns):
    client_data = df[df['ID client'] == client_id].drop(['Offre attribuée', 'ID client'], axis=1)

    # Convertir les données en variables numériques
    client_data = pd.get_dummies(client_data)
    client_data = client_data.reindex(columns=X_columns, fill_value=0)

    # Normaliser les données
    client_data = scaler.transform(client_data)

    # Faire la prédiction
    predicted_offer = clf.predict(client_data)

    return predicted_offer[0]

# Fonction principale pour orchestrer les différentes étapes
def main():
    # Chemin du fichier CSV
    file_path = 'clients_avec_offres (15).csv'
    
    # Charger et préparer les données
    X, y, df = charger_et_preparer_donnees(file_path)
    
    # Diviser et normaliser les données, capturer l'index de X_test
    X_train, X_test, y_train, y_test, scaler, X_test_index = diviser_normaliser_donnees(X, y)
    
    # Entraîner le modèle
    clf = entrainer_modele(X_train, y_train)
    
    # Optionnel : optimiser le modèle
    # clf = optimiser_modele(clf, X_train, y_train)
    
    # Évaluer le modèle
    evaluer_modele(clf, X_test, y_test)
    
    # Sauvegarder le modèle
    sauvegarder_modele(clf, scaler)
    
    # Faire une prédiction pour un client spécifique
    offre_predite = predire_offre('00-B8-91-CC', df, clf, scaler, X.columns)
    print(f"L'offre prédite pour le client 00-07-F3-D5 est : {offre_predite}")
    
    # Exemple de visualisation, en passant les index des tests
    visualiser_resultats(clf, X_test, y_test, df, X.columns, scaler, X_test_index)

# Fonction de visualisation
def visualiser_resultats(clf, X_test, y_test, df, X_columns, scaler, X_test_index):
    # Prédictions sur l'ensemble de test
    y_test_pred = clf.predict(X_test)
    
    # Conserver les IDs des clients avec l'index stocké avant la normalisation
    client_ids_test = df.loc[X_test_index, 'ID client']
    
    # Créer un DataFrame avec les résultats
    resultats = pd.DataFrame({
        'ID client': client_ids_test,
        'Offre réelle': y_test,
        'Offre prédite': y_test_pred
    })
    
    # Compter les valeurs réelles et prédites
    real_offer_counts = resultats['Offre réelle'].value_counts()
    predicted_offer_counts = resultats['Offre prédite'].value_counts()

    # Création du diagramme à barres comparatif
    labels = real_offer_counts.index
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, real_offer_counts, width=0.4, label='Offres réelles', align='center')
    ax.bar([i + 0.4 for i in x], predicted_offer_counts, width=0.4, label='Offres prédites', align='center')

    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Nombre d\'offres')
    ax.set_title('Comparaison des offres réelles et prédites')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Exécuter le programme principal
if __name__ == '__main__':
    main()
