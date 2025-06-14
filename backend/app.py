from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot.model import ChatbotModel
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the chatbot model
chatbot = ChatbotModel()

# Check if model exists, if not train it
if not os.path.exists('chatbot_model.h5'):
    print("Training new model...")
    train_x, train_y = chatbot.preprocess_data()
    chatbot.build_model(train_x, train_y)
    chatbot.train_model(train_x, train_y)
    chatbot.save_model()
else:
    print("Loading existing model...")
    chatbot.load_model()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 