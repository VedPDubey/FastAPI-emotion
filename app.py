import uvicorn
from fastapi import FastAPI
import socket

from log import Log

import pandas as pd
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,load_model

model = load_model('cnn_w2v.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = FastAPI()

@app.post('/predict')
def predict_wellness(data:Log):
    data=data.dict()
    message=[data['journalEntry']]
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=500)
    
    class_names = ['joy', 'anxiety', 'anger', 'sadness', 'neutral']

    pred = model.predict(padded)

    score = np.amax(pred)

    emotion = class_names[np.argmax(pred)]

    if(emotion=='sadness'):
        score=1+score
    elif(emotion=='anxiety'):
        score=2+score
    elif(emotion=='anger'):
        score=3+score
    elif(emotion=='neutral'):
        score=4+score
    elif(emotion=='joy'):
        score=5+score

    return {
        "emotion":emotion,
        "score":round(score,2)
    }

if __name__=='__main__':
    uvicorn.run(app,host=socket.gethostname(),port=8000)