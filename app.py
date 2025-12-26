import numpy as np
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import pandas as pd
import warnings

# Supprimer les avertissements de TensorFlow/Keras pour un environnement propre
warnings.filterwarnings("ignore")

# --- Paramètres globaux (DOIVENT correspondre à l'entraînement !) ---
MAX_LEN = 500       
MODEL_PATH = 'best_lstm_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

# --- 1. CHARGEMENT DES RESSOURCES GLOBALES ---
try:
    # Chargement du Modèle et du Tokenizer
    loaded_model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    # Initialisation des outils de Prétraitement (identique à nlp_preprocess_final_V2)
    lemmatizer = WordNetLemmatizer()
    STOP_WORDS = set(stopwords.words('english'))
    # Ajout des termes personnalisés retirés précédemment
    custom_removals = {'mr', 'u', 'sep', '[sep]'} 
    STOP_WORDS.update(custom_removals)
    # Ajout des lettres uniques
    STOP_WORDS.update(list('abcdefghijklmnopqrstuvwxyz')) 

    print("✅ Ressources de production chargées avec succès.")

except Exception as e:
    print(f"❌ ERREUR CRITIQUE : Échec du chargement des ressources. ({e})")
    print("Veuillez vérifier les chemins des fichiers .h5 et .pickle.")
    # On sort du script si les fichiers essentiels manquent
    exit()

# --- 2. FONCTION DE PRÉTRAITEMENT DE PRODUCTION (Cœur du Pipeline) ---
def preprocess_for_lstm(text):
    """Applique la séquence exacte de nettoyage, tokenisation et padding."""
    
    # 2a. Nettoyage et Lemmatisation
    text = str(text).strip()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = []
    
    for word in tokens:
        word = word.strip().lower() # Nettoyage et mise en minuscule
        
        if word in STOP_WORDS:
            continue
        if word.isdigit():
            continue
        if len(word) <= 1:
            continue
            
        lemmas = lemmatizer.lemmatize(word)
        filtered_tokens.append(lemmas)
        
    final_text = " ".join(filtered_tokens)

    # 2b. Tokenisation Keras
    sequence = loaded_tokenizer.texts_to_sequences([final_text])

    # 2c. Padding
    padded_sequence = pad_sequences(
        sequence, 
        maxlen=MAX_LEN, 
        padding='post', 
        truncating='post'
    )
    
    return padded_sequence

# --- 3. FONCTION DE PRÉDICTION FINALE (Interface API) ---
def predict_news_type(article_text):
    """Prédit si un article est FAKE ou REAL et retourne le résultat formaté."""
    
    if not article_text or len(article_text.strip()) == 0:
        return "Erreur : Veuillez entrer un texte.", 0.0

    # Prétraitement et Séquençage
    input_data = preprocess_for_lstm(article_text)
    
    # Prédiction de la probabilité
    probability = loaded_model.predict(input_data, verbose=0)[0][0]
    
    # Détermination de la classe (seuil 0.5)
    if probability >= 0.5:
        prediction = "REAL News"
        confidence = probability * 100
    else:
        prediction = "FAKE News"
        confidence = (1 - probability) * 100
    confidence_python_float = float(confidence)
        
    return prediction, confidence_python_float


###########partie flask apii##########
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Simuler le chargement des dépendances ici si vous utilisez des fichiers séparés ---
# NOTE: Dans un environnement réel, les fonctions de production_setup sont chargées au démarrage.
# Pour le test, assurez-vous que les variables (loaded_model, loaded_tokenizer, predict_news_type) sont accessibles ici.

# --- Route de la Page d'Accueil (pour l'interface utilisateur) ---
@app.route('/')
def index():
    # Vous devrez créer un fichier HTML simple (e.g., templates/index.html) pour l'interface
    return render_template('index.html') 

# --- Route de l'API de Prédiction (Gère la logique) ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Réception du texte
        data = request.get_json()
        article_text = data.get('text', '')
        
        if not article_text or len(article_text.strip()) == 0:
            return jsonify({"error": "Veuillez fournir un article à analyser."}), 400
        
        # 2. Appel de la fonction de prédiction (du script production_setup)
        # Si vous avez mis les fonctions dans le même fichier, l'appel est direct.
        prediction, confidence = predict_news_type(article_text)
        
        # 3. Retourne le résultat formaté
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "status": "success"
        })

    except Exception as e:
        # En cas d'erreur de traitement
        return jsonify({"error": f"Erreur de traitement serveur : {e}"}), 500

if __name__ == '__main__':
    print("Démarrage du serveur Flask... L'API sera disponible sur http://127.0.0.1:5000")
    # En environnement local, Flask démarre le serveur.
    app.run(debug=True)