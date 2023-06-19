from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('modelo.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Função para carregar e validar uma única imagem
def validar_imagem(caminho_imagem, modelo):
    img = image.load_img(caminho_imagem, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
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
