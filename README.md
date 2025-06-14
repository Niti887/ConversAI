# ConversAI - Smart Customer Service Chatbot

ConversAI is an intelligent chatbot designed to handle customer service queries using natural language processing and machine learning. It uses a neural network to understand user intents and provide appropriate responses.

## Features

- Natural language understanding using NLTK and TensorFlow
- Intent-based response system
- Modern web interface
- RESTful API for integration
- Easy to extend with new intents and responses

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd conversai
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Start chatting with the bot!

## Project Structure

```
conversai/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── data/
│   └── intents.json      # Training data
├── chatbot/
│   └── model.py          # Chatbot model implementation
└── templates/
    └── index.html        # Web interface
```

## Customization

### Adding New Intents

To add new intents, edit the `data/intents.json` file. Each intent should have:
- A unique tag
- A list of patterns (example user inputs)
- A list of responses

Example:
```json
{
    "tag": "new_intent",
    "patterns": [
        "Example pattern 1",
        "Example pattern 2"
    ],
    "responses": [
        "Response 1",
        "Response 2"
    ]
}
```

### Training the Model

The model is automatically trained when you first run the application. If you want to retrain the model, simply delete the `chatbot_model.h5` file and restart the application.

## API Usage

The chatbot can be accessed via a REST API:

```bash
curl -X POST http://localhost:5000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello"}'
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
