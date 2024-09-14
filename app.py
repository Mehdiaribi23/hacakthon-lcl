from flask import Flask, request, jsonify

# Initialisation de l'application Flask
app = Flask(__name__)

# Fonction de prédiction exemple (à remplacer par votre logique)
def predire_offre(client_id):
    # Logique simple pour illustrer
    if client_id == "00-07-F3-D5":
        return "Offre A"
    else:
        return "Offre B"

# Route de l'API pour recevoir un ID client et retourner l'offre prédite
@app.route('/predire', methods=['POST'])
def predict():
    data = request.get_json()  # Récupérer les données JSON envoyées
    client_id = data.get('client_id')  # Extraire l'id client du JSON

    # Appel de la fonction de prédiction
    offre_predite = predire_offre(client_id)

    # Retourner la réponse sous forme de JSON
    return jsonify({"client_id": client_id, "offre_predite": offre_predite})

# Lancer le serveur Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
