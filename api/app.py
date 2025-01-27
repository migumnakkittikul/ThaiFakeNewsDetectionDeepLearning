from flask import Flask, request, jsonify
from flask_cors import CORS  
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import re
from model_def import SimpleBiLSTM 

app = Flask(__name__)

CORS(app) 

device = torch.device('cpu')

with open('fake_news_model.pth', 'rb') as f:
    saved_package = torch.load(f, map_location=device, weights_only=False)

vocab_size = saved_package['vocab_size']
maxlen = saved_package['maxlen']
tokenizer = saved_package['tokenizer']

model = SimpleBiLSTM(vocab_size=vocab_size, emb_dim=200, n_eng_feats=4).to(device)
model.load_state_dict(saved_package['model_state_dict'])
model.eval()

def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'[^0-9a-zA-Z\u0E00-\u0E7F\s]', '', text)
    return text.strip()

def process_thai_text(text):
    text = clean_text(text)
    try:
        words = word_tokenize(text, engine='newmm')
        s = set(thai_stopwords())
        words = [w for w in words if w not in s and len(w) > 1]
    except:
        words = text.split()
    return ' '.join(words)

def add_engineered_features(text):
    text = clean_text(text)
    r = len(re.findall(r'[\u0E00-\u0E7F]', text)) / (len(text)+1)
    u = len(re.findall(r'http\S+|www\S+|https\S+', text))
    l = len(text)
    try:
        tw = len([w for w in word_tokenize(text, engine='newmm') if re.match(r'[\u0E00-\u0E7F]+', w)])
    except:
        tw = 0
    return [r, u, l, tw]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '')

    print("Received text:", text)  # used for debug

    text_clean = process_thai_text(text)
    print("Cleaned text:", text_clean)  # used for debug

    seq = tokenizer.texts_to_sequences([text_clean])
    print("Sequence:", seq)  # used for debug

    seq_pad = pad_sequences(seq, maxlen=maxlen, padding='post')
    print("Padded Sequence:", seq_pad)  # used for debug

    X_input = torch.tensor(seq_pad, dtype=torch.long).to(device)
    print("X_input Tensor:", X_input)  # used for debug

    eng_feat = add_engineered_features(text)
    print("Engineered Features:", eng_feat)  # used for debug

    eng_feat = torch.tensor(eng_feat, dtype=torch.float).unsqueeze(0).to(device)
    print("Eng_feat Tensor:", eng_feat)  # used for debug

    with torch.no_grad():
        output = model(X_input, eng_feat)
        print("Model Output:", output)  # used for debug
        pred = torch.argmax(output, dim=1).item()
        print("Predicted Class Index:", pred)  # used for debug

    label_map = {0: 'Fact News', 1: 'Fake News'}
    prediction = label_map.get(pred, 'Unknown')
    print("Final Prediction:", prediction)  # used for debug

    return jsonify({'prediction': prediction})

@app.route('/', methods=['GET'])
def home():
    return "Fake News Detection API is running!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
