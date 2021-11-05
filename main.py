from predict import get_similar_list
from flask import Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    pass
    return jsonify({'result': 1})