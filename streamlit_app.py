import os
import pickle
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import gdown

from model import ImageCaptioningModel, Vocabulary  # Your model.py should define these

# ==== Download Model from Google Drive ====
MODEL_ID = '1PBA8_U_vMymKMqCBgDAfTKme7WDJRVjs'  # Replace with your file ID
MODEL_PATH = 'model.pth'
VOCAB_PATH = 'vocab.pkl'

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("⏳ Downloading model from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
        st.write("✅ Model downloaded.")

download_model()

# ==== Load Vocabulary ====
@st.cache_data
def load_vocab():
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

vocab = load_vocab()

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Image Transform ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==== Caption Generation Function ====
def generate_caption(model, image, vocab, device, max_length=20):
    model.eval()
    result = []
    with torch.no_grad():
        features = model.encoder(image)
        caption = [vocab.stoi["<start>"]]

        for _ in range(max_length):
            cap_tensor = torch.tensor(caption).unsqueeze(0).to(device)
            output = model.decoder(features, cap_tensor)
            predicted = output.argmax(2)[:, -1].item()

            if predicted == vocab.stoi["<end>"]:
                break

            result.append(predicted)
            caption.append(predicted)

    # Remove consecutive duplicate words and special tokens
    final_words = []
    previous_word = None
    for idx in result:
        word = vocab.itos.get(idx, "<unk>")
        if word not in {"<start>", "<end>", "<pad>"}:
            if word != previous_word:
                final_words.append(word)
            previous_word = word
    return ' '.join(final_words)

# ==== Streamlit UI ====
st.title("🖼️ Générateur de Légendes d'Images")

uploaded_image = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Image sélectionnée", use_container_width=True)

        st.write("⏳ Chargement du modèle...")

        # Instantiate and load the model
        model = ImageCaptioningModel(
            embed_size=256,
            hidden_size=512,
            vocab_size=len(vocab)
        ).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        # Prepare image
        image_tensor = transform(image).unsqueeze(0).to(device)

        st.write("⏳ Génération de la légende...")
        caption = generate_caption(model, image_tensor, vocab, device)
        st.success(f"📜 Légende générée : **{caption}**")

    except Exception as e:
        st.error(f"Erreur lors du traitement de l’image : {e}")

# Debug: Show model file presence/size
st.write("Model exists:", os.path.exists(MODEL_PATH))
st.write("Model size:", os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else "Not found")
