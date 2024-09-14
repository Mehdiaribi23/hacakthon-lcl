import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import joblib 

# Charger les données et le modèle
file_path = 'clients_avec_offres (15).csv'
df = pd.read_csv(file_path)
scaler = StandardScaler()

# Charger le modèle avec joblib
model = joblib.load('modele_random_forest.pkl')

# Prétraiter les données (similaire à ce que vous avez dans client.py)
X = df.drop(['Offre attribuée'], axis=1)  # Variables explicatives
y = df['Offre attribuée']  # Cible

# Convertir les données catégoriques si nécessaire
X = pd.get_dummies(X)

# Recharger et appliquer la normalisation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fonction de prédiction
def predire_offre(client_id):
    # Filtrer les caractéristiques du client
    client_data = df[df['ID client'] == client_id].drop(['Offre attribuée', 'ID client'], axis=1)

    # Convertir les variables catégoriques en numériques
    client_data = pd.get_dummies(client_data)

    # Assurez-vous que les colonnes du client correspondent au jeu d'entraînement
    client_data = client_data.reindex(columns=X.columns, fill_value=0)

    # Appliquer la normalisation après avoir aligné les colonnes
    client_data = scaler.transform(client_data)

    # Faire la prédiction
    predicted_offer = model.predict(client_data)

    # Afficher la prédiction dans le terminal
    print(f"Prédiction pour le client {client_id} : {predicted_offer[0]}")

    # Retourner le résultat
    return predicted_offer[0]

# Tester la fonction avec un client spécifique
if __name__ == '__main__':
    # Vous pouvez tester avec un ID client existant
    test_client_id = '00-07-F3-D5'
    offre_predite = predire_offre(test_client_id)
    print(f"L'offre prédite pour le client {test_client_id} est : {offre_predite}")