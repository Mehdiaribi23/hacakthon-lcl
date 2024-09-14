from flask import Flask, request, jsonify
from client import charger_modele, charger_et_preparer_donnees, predire_offre

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger les données, le modèle et le scaler une fois au démarrage de l'application
file_path = 'clients_avec_offres (15).csv'
X, y, df = charger_et_preparer_donnees(file_path)  # Charger les données
clf, scaler = charger_modele('modele_random_forest.pkl', 'scaler.pkl')  # Charger le modèle et le scaler

# Route de l'API pour recevoir un ID client et retourner l'offre prédite
@app.route('/predire', methods=['POST'])
def predict():
    data = request.get_json()  # Récupérer les données JSON envoyées
    client_id = data.get('client_id')  # Extraire l'id client du JSON

    if not client_id:
        return jsonify({"error": "ID client manquant"}), 400

    try:
        # Appel de la fonction de prédiction dans client.py
        offre_predite = predire_offre(client_id, df, clf, scaler, X.columns)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Retourner la réponse sous forme de JSON
    return jsonify({"client_id": client_id, "offre_predite": offre_predite})

# Lancer le serveur Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
