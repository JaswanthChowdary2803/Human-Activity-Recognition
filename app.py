
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_text as text
import numpy as np
import speech_recognition as sr
import tensorflow_hub as hub
from flask import render_template
app = Flask(__name__)

# Load pre-trained BERT model from TensorFlow Hub
bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")

# Define a custom Keras layer for BERT preprocessing
class BERTPreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return bert_preprocess(inputs)

# Define a custom Keras layer for BERT encoding
class BERTEncoderLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return bert_encoder(inputs)

#print("Before loading model")
loaded_model = tf.keras.models.load_model(
    'my_BERT_modelfinal.h5',
    custom_objects={
        'BERTPreprocessLayer': BERTPreprocessLayer,
        'BERTEncoderLayer': BERTEncoderLayer
    }
)
print("After loading model")

# Add similar print statements in other critical sections


@app.route('/')
def index():
    return render_template('index.html')

def analyze_text_sentiment(input_text):
    print('jashu0')
    prediction = loaded_model.predict([input_text])   
    print('jashu1')
    return prediction

def analyze_audio_sentiment(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        input_text = recognizer.recognize_google(audio)
        prediction = analyze_text_sentiment(input_text)
        return prediction
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Error accessing the speech recognition service: {e}"


@app.route('/analyze_input', methods=['POST'])

def analyze_input():
    input_type = request.form['input_type']
    print("Hello")  
    if input_type == 'text':
        input_text = request.form['input_text']
        prediction = analyze_text_sentiment(input_text)
    elif input_type == 'audio':
        audio_file = request.files['audio_file']
        prediction = analyze_audio_sentiment(audio_file)
    print(prediction)
    
    out = ""
    if prediction[0][0]>0.7:
        out = "The Sentiment detected is Not Offence"
    else:
        out = "The Sentiment detected is Offence"
    return render_template('output.html', sentiment_result=out)





if __name__ == '__main__':
    app.run(debug=False)