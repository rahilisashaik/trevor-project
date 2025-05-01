import sys
import json
import librosa
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
import torchaudio
from torchaudio.transforms import MelSpectrogram
import soundfile as sf
import whisper
from scipy.io import wavfile
import tempfile

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

def analyze_sentiment_time_series(audio_path, chunk_size=10):
    """
    Analyze sentiment in audio chunks using both text and audio features.
    Args:
        audio_path: Path to the audio file
        chunk_size: Size of each chunk in seconds
    Returns:
        List of dictionaries containing sentiment scores and timestamps
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Calculate number of samples per chunk
    samples_per_chunk = sr * chunk_size
    
    # Initialize Whisper model for transcription
    whisper_model = whisper.load_model("base")
    
    # Initialize sentiment analyzer for text
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis",
        tokenizer="finiteautomata/bertweet-base-sentiment-analysis"
    )
    
    # Split audio into chunks
    num_chunks = int(np.ceil(len(y) / samples_per_chunk))
    sentiment_scores = []
    
    for i in range(num_chunks):
        start_sample = i * samples_per_chunk
        end_sample = min((i + 1) * samples_per_chunk, len(y))
        
        # Extract chunk
        chunk = y[start_sample:end_sample]
        
        # Save chunk temporarily for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, chunk, sr)
            # Transcribe chunk
            result = whisper_model.transcribe(temp_file.name)
            chunk_text = result["text"].strip()
        
        # Get text-based sentiment
        if chunk_text:
            sentiment_result = sentiment_analyzer(chunk_text)[0]
            # Convert sentiment to score between -1 and 1
            text_score = {
                'POS': 1.0,
                'NEU': 0.0,
                'NEG': -1.0
            }.get(sentiment_result['label'], 0.0) * sentiment_result['score']
        else:
            text_score = 0.0
        
        # Calculate audio features
        rms = librosa.feature.rms(y=chunk)[0]
        zcr = librosa.feature.zero_crossing_rate(chunk)[0]
        pitch, _ = librosa.piptrack(y=chunk, sr=sr)
        
        # Calculate audio-based features
        avg_rms = np.mean(rms)
        avg_zcr = np.mean(zcr)
        avg_pitch = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
        
        # Normalize audio features
        normalized_rms = (avg_rms - 0.02) / 0.04  # Typical RMS range
        normalized_zcr = (avg_zcr - 0.1) / 0.2    # Typical ZCR range
        normalized_pitch = (avg_pitch - 100) / 200 # Typical pitch range
        
        # Combine text and audio features
        # Weight text sentiment more heavily (0.7) than audio features (0.3)
        audio_score = (normalized_rms - normalized_zcr + normalized_pitch) / 3
        combined_score = 0.8 * text_score + 0.2 * audio_score
        
        # Ensure final score is between -1 and 1
        final_score = np.clip(combined_score, -1, 1)
        
        sentiment_scores.append({
            "timestamp": i * chunk_size,
            "score": float(final_score),
            "text": chunk_text,
            "audio_features": {
                "rms": float(avg_rms),
                "zcr": float(avg_zcr),
                "pitch": float(avg_pitch)
            }
        })
    
    return sentiment_scores

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sentiment_analysis.py <audio_file_path>", file=sys.stderr)
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    # Load model and feature extractor for emotion analysis
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
    id2label = model.config.id2label
    
    # Get emotion predictions
    predicted_emotion, predicted_strength = predict_emotion(audio_path, model, feature_extractor, id2label)
    
    # Calculate min and max weights for normalization
    minWeight = round(min(predicted_strength), 2)
    maxWeight = round(max(predicted_strength), 2)
    
    # Create emotion scores dictionary
    emotion_scores = {}
    for i in range(len(id2label)):
        normalized_score = float((round(predicted_strength[i], 2) - minWeight)/(maxWeight - minWeight))
        emotion_scores[predicted_emotion[i]] = normalized_score
    
    # Get time series sentiment analysis
    sentiment_time_series = analyze_sentiment_time_series(audio_path)
    
    # Combine both results into a single JSON object
    result = {
        "emotion_scores": emotion_scores,
        "sentiment_time_series": sentiment_time_series
    }
    
    # Print the combined results as JSON
    print(json.dumps(result))