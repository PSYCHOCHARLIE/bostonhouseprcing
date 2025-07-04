# import json
# import pickle

# from flask import Flask,request,app,jsonify,url_for,render_template
# import numpy as np
# import pandas as pd

# app=Flask(__name__)
# ## Load the model
# regmodel=pickle.load(open('regmodel.pkl','rb'))
# scalar=pickle.load(open('scaling.pkl','rb'))
# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

# @app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     final_input=scalar.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=regmodel.predict(final_input)[0]
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



# if __name__=="__main__":
#     app.run(debug=True)
   
     
import json
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler with error handling
try:
    regmodel = pickle.load(open('regmodel.pkl', 'rb'))
    scalar = pickle.load(open('scaling.pkl', 'rb'))
    print("Model and scaler loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    regmodel = None
    scalar = None

# Home route (GET) — renders the HTML form
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# API endpoint — receives JSON and returns prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Check if model is loaded
        if regmodel is None or scalar is None:
            return jsonify({"error": "Model not loaded properly"}), 500
        
        # Check if request contains JSON
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        # Get JSON data
        json_data = request.get_json()
        
        # Check if 'data' key exists
        if 'data' not in json_data:
            return jsonify({"error": "Missing 'data' key in JSON"}), 400
        
        data = json_data['data']
        print(f"Received data: {data}")
        
        # Convert to numpy array
        input_data = np.array(list(data.values())).reshape(1, -1)
        print(f"Input data shape: {input_data.shape}")
        
        # Scale the input
        scaled_input = scalar.transform(input_data)
        print(f"Scaled input: {scaled_input}")
        
        # Make prediction
        output = regmodel.predict(scaled_input)
        print(f"Prediction output: {output[0]}")
        
        return jsonify({
            "prediction": float(output[0]),
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in predict_api: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Web form submission route — receives form data and returns result in HTML
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if regmodel is None or scalar is None:
            return render_template("home.html", prediction_text="Error: Model not loaded properly")
        
        data = [float(x) for x in request.form.values()]
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        output = regmodel.predict(final_input)[0]
        return render_template("home.html", prediction_text=f"The House price prediction is {output:.2f}")
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": regmodel is not None})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)