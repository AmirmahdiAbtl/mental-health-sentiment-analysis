from fastapi import FastAPI, HTTPException
import numpy as np
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the pre-trained model
model = load_model('CNN model v3.h5')

# Class names for the predictions
classes_names = ['Anxiety', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'addiction', 'adhd']

# Load the tokenizer from a JSON file
with open('tokenizer.json') as json_file:
    tokenizer_json = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Sleep Issues model API"}

@app.post("/predict/")
def predict(data: dict):
    try:
        # Ensure input data contains the 'features' key
        if 'features' not in data:
            raise HTTPException(status_code=400, detail="Input data must contain 'features' key.")

        # Convert input text to sequences
        new_sequences = tokenizer.texts_to_sequences(data['features'])

        # Pad sequences to ensure consistent input size for the model
        new_padded = pad_sequences(new_sequences, maxlen=100, padding='post', truncating='post')

        # Make predictions using the model
        predictions = model.predict(new_padded)

        # Convert predictions to a JSON-serializable format
        predictions_list = predictions.tolist()  # Convert NumPy array to list

        # Map predictions to their corresponding class names (if needed)
        predicted_classes = [classes_names[np.argmax(pred)] for pred in predictions]

        return {
            "predictions": predictions_list,
            "predicted_classes": predicted_classes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
