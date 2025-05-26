
🚀 Projet de Génération Automatique de Légendes d'Images 🖼️

📚 Présentation
Ce dépôt contient un projet complet permettant la génération automatique de légendes d'images en utilisant des techniques avancées de deep learning. Le modèle proposé repose sur une architecture combinant un encodeur CNN pré-entraîné (ResNet18) 🧠 pour l'extraction des caractéristiques visuelles, et un décodeur RNN de type LSTM 📜 pour la génération textuelle. Le projet intègre également une interface web interactive développée avec Streamlit 🌐, ainsi qu'un notebook Jupyter 📒 détaillant le processus complet d'entraînement, d'évaluation et d'utilisation du modèle.

✨ Fonctionnalités principales
- 🖼️ Input an image via the GUI
- 🧠 Generate a caption using the trained model
- 💬 Display the generated caption under the image

📂 Contenu du dépôt

1. 🐍 model.py
🔍 Description détaillée :
- EncoderCNN 🖥️ : Utilise ResNet18 pré-entraîné.
- DecoderRNN 📝 : Réseau LSTM générateur de texte.
- Vocabulary 📖 : Gestion du vocabulaire.
- FlickrDataset 📷 : Chargement et traitement des données.

📌 Dépendances : PyTorch, Torchvision, Pillow

2. 🌐 streamlit_app.py
🔍 Application web Streamlit :
- 📂 Chargement d'images
- ⚡ Génération de légendes
- 🖼️ Affichage de la légende

🚀 Utilisation :
streamlit run streamlit_app.py

📌 Dépendances : Streamlit, PyTorch, Torchvision, Pillow

3. 📒 image_captioning.ipynb
🔍 Notebook Jupyter :
- 📥 Prétraitement Flickr8k
- 🛠️ Construction du modèle
- 🔬 Entraînement & visualisation
- 📊 Évaluation & tests

📌 Dépendances : Jupyter, PyTorch, Torchvision, Pillow, Matplotlib

🔧 Installation :
pip install torch torchvision streamlit pillow matplotlib

▶️ Utilisation rapide :
git clone https://github.com/votre-utilisateur/projet-generation-legendes.git
cd projet-generation-legendes
streamlit run streamlit_app.py

📸 Dataset :
Flickr8k, 8000 images avec légendes multiples.

🤝 Contributions :
Pull requests bienvenues !

👨‍💻 Auteurs :
- Youssef IDRISSI
- Souhail HAMIDI

📜 Licence :
MIT License
