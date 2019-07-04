from flask import Flask, request, render_template, jsonify
import numpy as np
import modelClass

app = Flask(__name__)

model = modelClass.NeuralNet('./model/model.json', './model/model.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def pr():
    msg = request.get_json(force=True)
    ins = []
    ins.append(float(msg['sep_len']))
    ins.append(float(msg['sep_wid']))
    ins.append(float(msg['pet_len']))
    ins.append(float(msg['pet_wid']))
    ins = np.array(ins)
    out = model.make_prediction(ins)
    print(out)
    response = {
        'spicies': out
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="localhost")