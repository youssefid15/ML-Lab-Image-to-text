import os
import pickle
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import gdown

from model import ImageCaptioningModel, Vocabulary

# --- Download Model ---
MODEL_URL = 'https://drive.google.com/uc?id=1PBA8_U_vMymKMqCBgDAfTKme7WDJRVjs'
MODEL_PATH = 'model.pth'
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model file...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.write("Model downloaded.")
download_model()

VOCAB_PATH = 'vocab.pkl'

# --- Load Vocab ---
def load_vocab():
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

# --- Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Caption Generation Function (your generate_caption) ---

# --- Streamlit App Interface ---
st.title("🖼️ Générateur de Légendes d'Images")
uploaded_image = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Image sélectionnée", use_container_width=True)
        st.write("⏳ Chargement du vocabulaire et du modèle...")
        
        vocab = load_vocab()
        model = ImageCaptioningModel(
            embed_size=256,
            hidden_size=512,
            vocab_size=len(vocab)
        ).to(device)
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.eval()
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        st.write("⏳ Génération de la légende...")
        caption = generate_caption(model, image_tensor, vocab, device)
        st.success(f"📜 Légende générée : **{caption}**")

    except Exception as e:
        st.error(f"Erreur lors du traitement de l’image : {e}")
