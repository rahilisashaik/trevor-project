import sys
import json
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torchaudio
from torchaudio.transforms import MelSpectrogram
import soundfile as sf

def remove_silence(audio_path, top_db=30):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Split the audio into non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # Concatenate non-silent intervals
    y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
    
    # Save the trimmed audio
    trimmed_path = audio_path.replace('.wav', '_trimmed.wav')
    sf.write(trimmed_path, y_trimmed, sr)
    
    return trimmed_path

def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    
    # Checks if GPU is available -> moves model to selected device that is available, moves inputs (preprocessed audio) to same device if true
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Disable gradient tracking -> making logit tensors
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    
    top_values, top_indices = torch.topk(logits, k = len(logits[0]))
    
    top = np.array(top_values[0])
    
    indices = np.array(top_indices[0])
    
    values = [0]*len(logits[0])
    
    i = 0
    for index in indices:
        values[i] = id2label[index]
        i += 1
    
    for i in range(len(top)):
        top[i] = round(top[i], 2)
    
    return values, top

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sentiment_analysis.py <audio_file_path>", file=sys.stderr)
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    # Load model and feature extractor
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
    id2label = model.config.id2label
    
    # Get predictions
    predicted_emotion, predicted_strength = predict_emotion(audio_path, model, feature_extractor, id2label)
    
    # Calculate min and max weights for normalization
    minWeight = round(min(predicted_strength), 2)
    maxWeight = round(max(predicted_strength), 2)
    
    # Create emotion scores dictionary
    emotion_scores = {}
    for i in range(len(id2label)):
        normalized_score = float((round(predicted_strength[i], 2) - minWeight)/(maxWeight - minWeight))
        emotion_scores[predicted_emotion[i]] = normalized_score
    
    # Only print the JSON output
    print(json.dumps(emotion_scores)) 