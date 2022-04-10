import random
import json
import torch
from Classes.Model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem

def main():
    
    # Set-up
    FILE = r'data\processed\model.pth'
    data = torch.load(FILE)

    input_size = data['input_size']
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    all_words = data['all_words']
    tags = data['tags']
    model_state = data['model_state']
    
    with open(r'data\training_data\train.json') as f:
        intents = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    botName = "Albert"
    print(f"{botName}> Hello, I am {botName}, let's talk fitness!")

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        
        
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        
        output = model(X)
        _, predicted = torch.max(output, 1)
        print(predicted)
        tag = tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()] # type: ignore
        print(f"{botName}> {tag} ({prob:.2f})")
        
        # Confidence routing
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    print(f"{botName}> {random.choice(intent['responses'])}")
                    break
        else:
            print(f"{botName}> I'm sorry, I don't understand.")    
        
    pass          
        
if __name__ == "__main__":
    main()

