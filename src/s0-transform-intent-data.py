import json

def main():
    
    FILEPATH = r"data\raw\data_full_response_data.json"
    OUTPUT_FILEPATH = r"data\training_data\data_full_response_data_transformed.json"
    
    with open(FILEPATH, "r") as f:
        data = json.load(f)

    intent_dict = {}

    for item in data['train']:
        text, intent = item[0], item[1]
        if intent not in intent_dict:
            intent_dict[intent] = []
        intent_dict[f'{intent}'].append(text)
    
    training_json = {}
    training_list = []
    for key, values in intent_dict.items():
        training_dict = {}
        training_dict['tag'] = key
        training_dict['patterns'] = values
        friendly_name = key.replace('_', ' ').title()
        training_dict['friendly_name'] = friendly_name
        # There's a lot of these intents... So I think I'll hold off on proper responses for now
        training_dict['responses'] = [f"I can tell that this is a '{friendly_name}' intent, but I'm not configured to respond to this just yet..."]
        training_list.append(training_dict)
    
    training_json['intents'] = training_list
    
    with open(OUTPUT_FILEPATH, "w") as f:
        json.dump(training_json, f, indent=4)
        
    return

if __name__ == "__main__":
    main()