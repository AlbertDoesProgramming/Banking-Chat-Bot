import PySimpleGUI as sg

sg.theme('Topanga')

layout = [[sg.Multiline(size=(80, 20), reroute_stdout=True, echo_stdout_stderr=True)],
          [sg.MLine(size=(70, 5), key='-MLINE IN-', enter_submits=True, do_not_clear=False),
           sg.Button('SEND', bind_return_key=True), sg.Button('EXIT')]]

window = sg.Window('Chat Window', layout,
            default_element_size=(30, 2))

while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Show':
        # Update the "output" text element to be the value of "input" element
        window['-OUTPUT-'].update(values['-IN-'])

# botName = "Albert"
# print(f"{botName}> Hello, I am {botName}, let's talk fitness!")

# while True:
#     sentence = input("You: ")
#     if sentence == "quit":
#         break
    
#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)
    
#     output = model(X)
#     _, predicted = torch.max(output, 1)
#     print(predicted)
#     tag = tags[predicted.item()]
    
#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()] # type: ignore
#     print(f"{botName}> {tag} ({prob:.2f})")
    
#     # Confidence routing
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent['tag']:
#                 print(f"{botName}> {random.choice(intent['responses'])}")
#                 break
#     else:
#         print(f"{botName}> I'm sorry, I don't understand.")    
    

window.close()