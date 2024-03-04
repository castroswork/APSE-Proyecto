import pickle
from flask import Flask, jsonify, request
import numpy as np
import json

app = Flask(__name__)

# Load the trained scikit-learn models stored in pickle format
with open('travelModel.pkl', 'rb') as f:
    modelo_tiempo_viaje = pickle.load(f)

with open('deliveryModel.pkl', 'rb') as f:
    modelo_tiempo_entrega = pickle.load(f)

with open('le.pkl', 'rb') as f:
    labelEncoder = pickle.load(f)



# Endpoint for route prediction model
# Input is a json object with attribute time
@app.route('/predict_eta', methods=['POST'])
def predict_eta():
    data = request.get_json()  # Obtener los datos JSON de la petición
    time_features = np.array(data['time']).reshape(1, -1)  # Asegurar la forma correcta para el modelo
    prediccion = modelo_tiempo_viaje.predict(time_features)  # Realizar la predicción
    return jsonify({'prediction': prediccion[0].tolist()})  # Devolver la predicción como respuesta JSON


# Endpoint for load delivery endpoint.
# Input is a json object with attributes truckId and time
@app.route('/predict_delivery', methods=['POST'])
def predict_delivery():
    data = request.get_json()
    truck_id = np.array([data['truckId']])  # Extraer el ID del camión
    time_features = np.array(data['time']).reshape(1, -1)  # Extraer y dar forma a las características de tiempo
    # Transformar el ID del camión y combinarlo con las características de tiempo
    encoded_truck_id = labelEncoder.transform(truck_id)
    features = np.hstack((encoded_truck_id.reshape(-1, 1), time_features))
    prediccion = modelo_tiempo_entrega.predict(features)  # Realizar la predicción
    return jsonify({'prediction': prediccion[0].tolist()})  # Devolver la predicción como respuesta JSON

if __name__ == '__main__':
    app.run(debug=True, port =7777, host='0.0.0.0')