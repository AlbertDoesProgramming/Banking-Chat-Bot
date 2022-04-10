import json

def main():
    
    FILEPATH = r"data\raw\data_full_response_data.json"
    OUTPUT_FILEPATH = r"data\processed\data_full_response_data_transformed.json"
    
    with open(FILEPATH, "r") as f:
        data = json.load(f)

    for item in data['train']:
        text, intent = item[0], item[1]
        
    
    return

if __name__ == "__main__":
    main()