from flask import Flask, request, render_template, send_file
import os
import pickle
import time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import io
import base64
import networkx as nx

app = Flask(__name__)
UPLOAD_FOLDER = 'data/uploads'
RESULT_FOLDER = 'data/results'
MODEL_PATH = 'data/model.h5'  # Model path
TOKENIZER_PATH = 'data/tokenizer.pickle'  # Tokenizer path

def load_model_and_tokenizer():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def plot_label_distribution(predicted_labels):
    """
    Plot distribution of predicted labels and return image in base64 encoding.
    """
    # Convert numpy array to DataFrame
    labels_df = pd.DataFrame(predicted_labels, columns=['anger','anticipation','disgust','fear','joy','sadness','surprise','trust','anies','prabowo','ganjar' ])  # Ganti Label_1, Label_2, dst dengan nama label Anda
    
    # Plot distribution
    label_counts = labels_df.sum(axis=0)
    plt.figure(figsize=(10, 5))
    label_counts.plot(kind='bar')
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    
    # Save plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    
    # Encode plot image to base64 to embed in HTML
    plot_url = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,' + plot_url

def social_network_analysis(data):
    # Memproses data untuk analisis jaringan
    # Misalnya, mengidentifikasi hubungan antara entitas dalam komentar
    
    # Contoh sederhana: membuat graf acak untuk tujuan demonstrasi
    G = nx.erdos_renyi_graph(len(data), 0.15)
    
    return G

def plot_social_network(data, file_name="sna_graph.png"):
    # Membuat graf
    G = nx.Graph()

    # Menambahkan nodes untuk tokoh politik dan emosi
    politicians = ['anies', 'prabowo', 'ganjar']
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

    # Menambahkan nodes dengan atribut
    for politician in politicians:
        G.add_node(politician, color='blue', size=700, type='politician')

    for emotion in emotions:
        G.add_node(emotion, color='red', size=500, type='emotion')

    # Menambahkan edges berdasarkan jumlah komentar yang mengasosiasikan tokoh dengan emosi tertentu
    for emotion in emotions:
        for politician in politicians:
            weight = data[data[politician] == 1][emotion].sum()
            if weight > 0:
                G.add_edge(politician, emotion, weight=weight)

    # Mengatur warna dan ukuran node
    node_colors = [G.nodes[node]['color'] for node in G]
    node_sizes = [G.nodes[node]['size'] for node in G]

    # Mengatur lebar garis berdasarkan bobot
    edge_widths = [edgedata['weight'] / 300 for _, _, edgedata in G.edges(data=True)]

    # Menggunakan layout yang lebih menarik
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    # Menggambar graf dengan visualisasi yang lebih baik
    plt.figure(figsize=(14, 12))
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, width=edge_widths,
            with_labels=True, font_size=12, font_color='white', edge_color='#FF5733')
    plt.title("Emotional Associations with Political Figures")

    # Simpan plot ke buffer bytes
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode plot image ke base64 untuk disematkan di HTML
    plot_url = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,' + plot_url

     # Save the graph as a PNG image in the RESULT_FOLDER
    file_path = os.path.join(RESULT_FOLDER, file_name)
    plt.savefig(file_path)
    plt.close()  # Close the figure to free memory

    return file_name

@app.route('/download/sna/<filename>')
def download_sna(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True, download_name="sna_graph.png")

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)
        
        label_distribution_plot, sna_plot, result_filename = process_and_plot(filepath)
        
        return render_template('results.html', label_distribution=label_distribution_plot, sna_plot=sna_plot, result_filename=result_filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

def process_and_plot(file_path):
    data = pd.read_excel(file_path)
    texts = data['comment_text'].astype(str).values
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_data = pad_sequences(sequences, maxlen=200)
    
    predictions = model.predict(padded_data)
    
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)
       
    # Save the predictions and texts to an Excel file
    data_with_predictions = data.copy()
    for idx, emotion in enumerate(['anger','anticipation','disgust','fear','joy','sadness','surprise','trust','anies','prabowo','ganjar']):
        data_with_predictions[emotion] = predicted_labels[:, idx]
    
    # Define the filename with a timestamp or unique identifier to avoid collisions
    result_filename = f"predictions_{int(time.time())}.xlsx"
    data_with_predictions.to_excel(os.path.join(RESULT_FOLDER, result_filename), index=False)

    # Generate label distribution plot
    label_distribution_plot = plot_label_distribution(predicted_labels)

    # Check if required columns are present in the data
    required_columns = ['anies', 'prabowo', 'ganjar', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    if all(col in data.columns for col in required_columns):
    # Perform social network analysis
        sna_plot_filename = plot_social_network(data)  # Update this line
    else:
        sna_plot_filename = None  # If required columns are not present, set sna_plot_filename to None

    return label_distribution_plot, sna_plot_filename, result_filename



if __name__ == '__main__':
    app.run(debug=True)
