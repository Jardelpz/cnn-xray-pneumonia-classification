import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps


app = Flask(__name__)
model = load_model('modelo.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Função para carregar e validar uma única imagem
def validar_imagem(caminho_imagem, modelo):
    # Carregar a imagem em escala de cinza
    img = load_img(caminho_imagem, color_mode='grayscale', target_size=(150, 150))
    # Converter a imagem para um array numpy
    img_array = img_to_array(img)
    # Expandir as dimensões do array para corresponder ao formato esperado pelo modelo
    img_array = np.expand_dims(img_array, axis=0)
    # Normalizar os valores de pixel no intervalo [0, 1]
    img_array /= 255.0

    # Fazer a previsão com o modelo
    resultado = modelo.predict(img_array)
    classe = "NORMAL" if resultado[0][0] < 0.5 else "PNEUMONIA"

    return classe


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Carregar o modelo treinado
        model = load_model('modelo.h5')

        file = request.files['file']
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        predict = validar_imagem(file_path, model)

        return render_template('result.html', result=predict, image_path=file_path)


if __name__ == '__main__':
    app.run(debug=True)
