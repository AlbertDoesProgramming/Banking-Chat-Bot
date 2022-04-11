from nltk_utils import tokenize, stem, bag_of_words
from Classes.ChatDataset import ChatDataset
from Classes.Model import NeuralNet
import numpy as np
import json
import re
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def main():
    
    def load_json_data(filepath):
        with open(filepath) as f:
            intentsData = json.load(f)
        return intentsData
    
    def extract_allwords_tags_xy(intentsData):
        allWords = []
        tags = []
        xy = []        
        for intent in intentsData['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                w = tokenize(pattern)
                allWords.extend(w)
                xy.append((w, tag))
        allWords = sorted(set([stem(word) for word in allWords if re.search('[a-zA-Z]', word)]))
        tags = sorted(set(tags))
        
        return allWords, tags, xy

    
    def get_XY_training_data(xy, allWords, tags):
        x_train, y_train = [], []
        for (pattern_sentence, tag) in xy:
            bag = bag_of_words(pattern_sentence, allWords)
            x_train.append(bag)            
            label = tags.index(tag)
            y_train.append(label) 
        return np.array(x_train), np.array(y_train)

    def get_hyper_params(tags, X_train, hyperPath):
        
        with open(hyperPath) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        batch_size = params['batch_size']
        hidden_size = params['hidden_size']
        output_size = len(tags)
        input_size = len(X_train[0])
        learning_rate = params['learning_rate']
        num_epochs = params['num_epochs']
        MODEL_FILE_PATH = params['MODEL_FILE_PATH']
        
        return batch_size, hidden_size, output_size, input_size, learning_rate, num_epochs, MODEL_FILE_PATH


    def train_model(X_train, Y_train, batch_size, hidden_size, output_size, input_size, learning_rate, num_epochs):
        
        dataset = ChatDataset(X_train, Y_train)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
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
                print(f'Loss: {loss.item()}:.4f') # type: ignore

        print(f'Final Loss: {loss.item()}:.4f') # type: ignore
        
        dataModel = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "all_words": allWords,
            "tags": tags
        }
        
        return dataModel 
    
    # load data
    paramsPath = r'src\params.yml'    
    #intents = load_json_data(r'data\training_data\train.json')
    intents = load_json_data(r'data\training_data\data_full_response_data_transformed.json')
    allWords, tags, xy = extract_allwords_tags_xy(intents)
    X_train, Y_train = get_XY_training_data(xy, allWords, tags)

    # Hyperparameters
    batch_size, hidden_size, output_size, input_size, learning_rate, num_epochs, MODEL_FILE_PATH = get_hyper_params(tags, X_train, paramsPath)
    
    # Train Model
    dataModel = train_model(X_train, Y_train, batch_size, hidden_size, output_size, input_size, learning_rate, num_epochs)
    torch.save(dataModel, MODEL_FILE_PATH)
    print(f'Model saved to {MODEL_FILE_PATH}')
    
if __name__ == '__main__':
    main()