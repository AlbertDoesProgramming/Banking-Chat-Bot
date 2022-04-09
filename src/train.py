from nltk_utils import tokenize, stem, bag_of_words
from ChatDataset import ChatDataset
from model import NeuralNet

import numpy as np
import json
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def main():
    with open(r'data\training_data\train.json') as f:
        intents = json.load(f)

    allWords = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            allWords.extend(w)
            xy.append((w, tag))

    allWords = sorted(set([stem(word) for word in allWords if re.search('[a-zA-Z]', word)]))
    tags = sorted(set(tags))

    x_train, y_train = [], []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, allWords)
        x_train.append(bag)
        
        label = tags.index(tag)
        y_train.append(label) 

    X_train = np.array(x_train)
    Y_train = np.array(y_train)

    # hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    print(input_size, len(allWords))
    print(output_size, tags)
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimser
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            
            output = model(words)
            loss = criterion(output, labels.long())
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        if (epoch +1) % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Loss: {loss.item()}:.4f')

    print(f'Final Loss: {loss.item()}:.4f')        
    
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": allWords,
        "tags": tags
    }        
    
    FILE = "data.pth"
    torch.save(data, FILE)
    print(f'Model saved to {FILE}')
    
if __name__ == '__main__':
    main()