import streamlit as st
import warnings
import torch
import json
from models import Encoder, DecoderWithAttention
from utils import caption_image_beam_search
import numpy as np
from PIL import Image
import io
import skimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gdown

# Filter out UserWarnings
warnings.filterwarnings("ignore")

# Download model checkpoint
file_id = '1dVucDQ9BuVwASl0ZH5Ba5LKflXMSi6W2'
output = 'BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar'
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output, quiet=False)

# model config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# output_path = ''
model_path = './BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar'
word_map_path = './WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json'
beam_size = 5
smooth = False

# Load word map (word2ix)
with open(word_map_path, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

# load decoder and encoder for loading the checkpoint, not for using them
decoder = DecoderWithAttention(attention_dim=512,
                                embed_dim=300,
                                decoder_dim=512,
                                vocab_size=len(word_map),
                                dropout=0.5)
encoder = Encoder()

# Load model
checkpoint = torch.load(model_path, map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

st.title("Image captioning with encoder cnn & decoder lstm with attension network")

uploaded_file = st.file_uploader("Upload an image to predict...", type="jpg")

if uploaded_file is not None:
    content = uploaded_file.read()
    st.image(content, caption="Uploaded Image.", use_column_width=True)
    img = np.array(Image.open(io.BytesIO(content)))

    # Encode, decode with attention and beam search
    best_seq, alphas, complete_seqs = caption_image_beam_search(encoder, decoder, img, word_map, device, beam_size)
    alphas = torch.FloatTensor(alphas)

    # Print all possible captions
    st.header("Predict captions:")
    i = 1
    for cap in complete_seqs:
        st.write(f'{i}.', ' '.join([rev_word_map[w] for w in cap[1:-1]]))

    # Visualize caption and attention of best sequence
    st.header("Best sequence with attension:")
    image = Image.fromarray(img)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in best_seq]
    num_words = min(len(words), 50)
    num_cols = 5
    num_rows = (num_words + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    axes = axes.flatten()
    
    for t in range(num_words):
        if t >= len(axes):
            break

        ax = axes[t]
        ax.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        ax.imshow(image)

        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24], anti_aliasing=True)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24], anti_aliasing=False)

        if t == 0:
            ax.imshow(alpha, alpha=0, cmap=cm.Greys_r)
        else:
            ax.imshow(alpha, alpha=0.8, cmap=cm.Greys_r)
        ax.axis('off')
    
    for t in range(num_words, len(axes)):
        axes[t].axis('off')
    
    st.pyplot(fig)
