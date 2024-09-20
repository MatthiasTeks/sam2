from flask import Blueprint, jsonify, request
from .client_management import instantiate_model, handle_ping, predict_for_client

main = Blueprint('main', __name__)

@main.route('/instantiate', methods=['POST'])
def instantiate():
    data = request.json
    client_id = data.get('client_id')
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400

    response = instantiate_model(client_id)
    return jsonify(response), 200

@main.route('/ping', methods=['POST'])
def ping():
    data = request.json
    client_id = data.get('client_id')
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400

    response = handle_ping(client_id)
    return jsonify(response), 200

@main.route('/predict', methods=['POST'])
def predict():
    data = request.json
    client_id = data.get('client_id')
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400

    image_base64 = data.get('image')
    if not image_base64:
        return jsonify({'error': 'Image non fournie'}), 400

    response = predict_for_client(client_id, image_base64)
    return jsonify(response), 200