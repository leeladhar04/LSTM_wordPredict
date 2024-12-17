from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model("next_word.keras")
tokenizer=pickle.load(open('token.pkl','rb'))


def predict_next_word(model,tokenizer,text):
    sequence=tokenizer.texts_to_sequences([text])
    sequence=np.array(sequence)
    preds=np.argmax(sequence)
    preds=np.argmax(model.predict(sequence))
    predicted_word=""
    for key,value in tokenizer.word_index.items():
        if value==preds:
            predicted_word=key
            break
    print(predicted_word)
    return predicted_word


predict_next_word(model,tokenizer,"I merely shared with all")
