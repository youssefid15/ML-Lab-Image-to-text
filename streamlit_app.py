import sys
import os
import pickle
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle

from model import ImageCaptioningModel, EncoderCNN, DecoderRNN, Vocabulary, FlickrDataset

import os
import requests

MODEL_URL = 'https://drive.google.com/file/d/1PBA8_U_vMymKMqCBgDAfTKme7WDJRVjs/view?usp=drive_link'
MODEL_PATH = 'model.pth'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")

# Call the function at the start of your app
download_model()

# CONFIG
#MODEL_PATH = "C:/Users/Idris/OneDrive/Desktop/ML PROJECT 3/model.pth"
MODEL_PATH = "model.pth"
IMAGE_DIR = "C:/Users/Idris/OneDrive/Desktop/ML PROJECT 3/Flickr8k_Dataset/Images"
CAPTIONS_FILE = "C:/Users/Idris/OneDrive/Desktop/ML PROJECT 3/Flickr8k_Dataset/captions.txt"
VOCAB_PATH = "vocab.pkl"
#VOCAB_PATH = "C:/Users/Idris/OneDrive/Desktop/ML PROJECT 3/vocab.pkl"
#MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
print("Taille du vocabulaire :", len(vocab))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"🔤 Taille du vocabulaire : {len(vocab)}")


# Prétraitement des images


# Code Streamlit – à corriger ainsi :
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Fonction pour charger captions.txt
def load_captions(captions_file_path):
    captions_dict = {}
    with open(captions_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            image_name, caption = parts
            image_name = image_name.split('#')[0]  # remove #0, #1 etc.
            if image_name not in captions_dict:
                captions_dict[image_name] = []
            captions_dict[image_name].append(caption)
    return captions_dict

# Fonction de chargement ou création du vocabulaire
@st.cache_data
def load_vocab():
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        return vocab
    else:
        captions_dict = load_captions(CAPTIONS_FILE)
        dataset = FlickrDataset(captions_dict, IMAGE_DIR, transform)
        vocab = dataset.vocab
        with open(VOCAB_PATH, 'wb') as f:
            pickle.dump(vocab, f)
        return vocab

# Génération de légende
def generate_caption(model, image, vocab, device, max_length=20):
    model.eval()
    result = []
    with torch.no_grad():
        features = model.encoder(image)
        caption = [vocab.stoi["<start>"]]

        for _ in range(max_length):  # ✅ ici on boucle sur max_length
            cap_tensor = torch.tensor(caption).unsqueeze(0).to(device)
            output = model.decoder(features, cap_tensor)
            predicted = output.argmax(2)[:, -1].item()

            if predicted == vocab.stoi["<end>"]:
                break

            result.append(predicted)
            caption.append(predicted)
    # Suppression des mots consécutifs répétés
    final_words = []
    previous_word = None
    for idx in result:
        word = vocab.itos.get(idx, "<unk>")
        if word not in {"<start>", "<end>", "<pad>"}:
            if word != previous_word:
                final_words.append(word)
            previous_word = word

    return ' '.join(final_words)

    return ' '.join([
    word for idx in result
    if (word := vocab.itos.get(idx, "<unk>")) not in {"<start>", "<end>", "<pad>"}
])




# === INTERFACE STREAMLIT ===
st.title("🖼️ Générateur de Légendes d'Images")

uploaded_image = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Image sélectionnée", use_container_width=True)

    st.write("⏳ Chargement du vocabulaire et du modèle...")

    vocab = load_vocab()

    model = ImageCaptioningModel(
        embed_size=256,
        hidden_size=512,
        vocab_size=len(vocab)
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Préparer l'image pour le modèle
    image_tensor = transform(image).unsqueeze(0).to(device)

    st.write("⏳ Génération de la légende...")
    caption = generate_caption(model, image_tensor, vocab, device)
    st.success(f"📜 Légende générée : **{caption}**")
   
