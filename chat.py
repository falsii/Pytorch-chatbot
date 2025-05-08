import random
import json
import logging
import re
from datetime import datetime
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Set up logging
logging.basicConfig(
    filename=f'chatbot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_intents(file_path='intents.json'):
    """Load intents from JSON file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as json_data:
            return json.load(json_data)
    except FileNotFoundError:
        logging.error(f"Intents file {file_path} not found")
        raise
    except json.JSONDecodeError:
        logging.error("Invalid JSON format in intents file")
        raise

def load_model_data(file_path="data.pth"):
    """Load model data with error handling"""
    try:
        return torch.load(file_path)
    except FileNotFoundError:
        logging.error(f"Model data file {file_path} not found")
        raise
    except Exception as e:
        logging.error(f"Error loading model data: {str(e)}")
        raise

def clean_input(text):
    """Clean and validate user input"""
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove potentially harmful characters
    text = re.sub(r'[<>|&;]', '', text)
    return text

def get_response(model, sentence, all_words, tags, intents, confidence_threshold=0.75):
    """Get chatbot response based on input sentence"""
    sentence = tokenize(sentence)
    if not sentence:
        return "Please say something!", 0.0

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > confidence_threshold:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Weighted response selection based on probability
                response = random.choices(
                    intent['responses'],
                    weights=[1.0, 0.8, 0.6, 0.4][:len(intent['responses'])],
                    k=1
                )[0]
                return response, prob.item()
    return "I'm not sure I understand. Could you rephrase?", prob.item()

def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load data
    try:
        intents = load_intents()
        data = load_model_data()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Initialize model
    try:
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        print("Error initializing chatbot")
        return

    bot_name = "Falsi"
    conversation_history = []
    max_history = 10  # Store last 10 exchanges

    print(f"{bot_name}: Let's chat! (type 'quit' to exit or 'history' to see conversation)")
    
    while True:
        try:
            user_input = input("You: ").lower()
            user_input = clean_input(user_input)

            if user_input == "quit":
                logging.info("Chat session ended by user")
                break
            elif user_input == "history":
                print("\nConversation History:")
                for exchange in conversation_history:
                    print(f"You: {exchange['user']}")
                    print(f"{bot_name}: {exchange['bot']} (Confidence: {exchange['confidence']:.2f})")
                continue
            elif not user_input:
                print(f"{bot_name}: Please type something!")
                continue

            response, confidence = get_response(
                model, user_input, all_words, tags, intents
            )
            
            # Store in history
            conversation_history.append({
                'user': user_input,
                'bot': response,
                'confidence': confidence
            })
            if len(conversation_history) > max_history:
                conversation_history.pop(0)

            print(f"{bot_name}: {response} (Confidence: {confidence:.2f})")
            logging.info(f"User: {user_input} | Bot: {response} | Confidence: {confidence:.2f}")

        except Exception as e:
            logging.error(f"Error processing input: {str(e)}")
            print(f"{bot_name}: Oops, something went wrong. Please try again!")

if __name__ == "__main__":
    main()
