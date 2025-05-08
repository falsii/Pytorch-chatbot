import numpy as np
import random
import json
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Set up logging
logging.basicConfig(
    filename=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_intents(file_path='intents.json'):
    """Load intents from JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Intents file {file_path} not found")
        raise
    except json.JSONDecodeError:
        logging.error("Invalid JSON format in intents file")
        raise

# Load intents
try:
    intents = load_intents()
except Exception as e:
    print(f"Error loading intents: {str(e)}")
    exit(1)

# Prepare data
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and preprocess words
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

logging.info(f"Loaded {len(xy)} patterns, {len(tags)} tags, {len(all_words)} unique stemmed words")

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
config = {
    'num_epochs': 1000,
    'batch_size': 8,
    'learning_rate': 0.001,
    'hidden_size': 8,
    'dropout_rate': 0.2,
    'patience': 50,
    'min_delta': 0.0001,
    'lr_factor': 0.5,
    'lr_patience': 20,
    'valid_split': 0.3  # Increased to ensure enough samples
}

input_size = len(X_train[0])
output_size = len(tags)
logging.info(f"Input size: {input_size}, Output size: {output_size}")

class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Check if validation split is feasible
min_valid_size = len(tags)  # Need at least one sample per class
valid_size = int(len(X_train) * config['valid_split'])
if len(X_train) < 10 or valid_size < min_valid_size:
    logging.warning(
        f"Dataset too small ({len(X_train)} samples) or validation size ({valid_size}) "
        f"less than number of classes ({min_valid_size}). Using all data for training."
    )
    train_dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    valid_loader = None
else:
    train_idx, valid_idx = train_test_split(
        range(len(X_train)),
        test_size=config['valid_split'],
        random_state=42,
        stratify=y_train
    )
    train_dataset = ChatDataset(X_train[train_idx], y_train[train_idx])
    valid_dataset = ChatDataset(X_train[valid_idx], y_train[valid_idx])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

model = NeuralNet(input_size, config['hidden_size'], output_size, config['dropout_rate']).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config['lr_factor'],
    patience=config['lr_patience'],
    verbose=True
)

# Training loop
best_loss = float('inf')
patience_counter = 0
best_model_path = "best_model.pth"

for epoch in range(config['num_epochs']):
    model.train()
    total_train_loss = 0
    for (words, labels) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation (if available)
    avg_valid_loss = float('inf')
    if valid_loader:
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for words, labels in valid_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                outputs = model(words)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)

    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

    # Save best model and early stopping (if validation available)
    if valid_loader:
        if avg_valid_loss < best_loss - config['min_delta']:
            best_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model with validation loss: {best_loss:.4f}")
        else:
            patience_counter += 1

        scheduler.step(avg_valid_loss)

        if patience_counter >= config['patience']:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Early stopping at epoch {epoch+1}")
            break
    else:
        # Save model periodically if no validation
        if avg_train_loss < best_loss - config['min_delta']:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model with train loss: {best_loss:.4f}")
        scheduler.step(avg_train_loss)

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{config['num_epochs']}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')

# Load best model for saving final data
try:
    model.load_state_dict(torch.load(best_model_path))
except FileNotFoundError:
    logging.warning("No best model found, saving current model state")

# Save final data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": config['hidden_size'],
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
try:
    torch.save(data, FILE)
    logging.info(f"Training complete. Final model saved to {FILE}")
    print(f"Training complete. File saved to {FILE}")
except Exception as e:
    logging.error(f"Error saving model: {str(e)}")
    print(f"Error saving model: {str(e)}")
