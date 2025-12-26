Ce projet utilise l'intelligence artificielle pour classifier automatiquement des articles de presse en deux catégories : Vrai (REAL) ou Faux (FAKE).

📝 Description
L'objectif est de fournir un outil capable d'analyser le style et le contenu d'un texte pour identifier les signaux de désinformation. 
Le projet couvre l'ensemble de la chaîne : de la collecte des données à la mise en ligne d'une interface de test.

🚀 Résultats Techniques
Le modèle final repose sur une architecture de Deep Learning performante :
•	Modèle : LSTM (Long Short-Term Memory).
•	Précision (Accuracy) : 96.66%.
•	F1-Score : 96.73%.

📊 Données utilisées
Le modèle a été entraîné sur un dataset hybride de plus de 72 000 articles :
•	Données extraites par scraping (Snopes).
•	Enrichissement via des datasets publics (Kaggle).
Classe,Nombre d'articles,Longueur Moyenne
FAKE (0),35 242,331 mots
REAL (1),37 254,286 mots

🛠️ Technologies
•	NLP : NLTK, Scikit-learn (TF-IDF).
•	Deep Learning : TensorFlow & Keras.
•	Web : Flask (pour l'interface de prédiction).



