import flask
from flask import Flask, request, jsonify
import pickle  # ou torch, tensorflow, selon votre modèle

# Charger votre modèle (par exemple un modèle pickle, TensorFlow, etc.)
# Exemple avec un modèle pickle
model = pickle.load(open('votre_modele.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Recevoir les données depuis une requête POST
    data = request.get_json()
    
    # Faire une prédiction avec le modèle
    prediction = model.predict([data['input']])  # Adapter selon votre modèle
    
    # Envoyer la prédiction comme réponse JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run()
