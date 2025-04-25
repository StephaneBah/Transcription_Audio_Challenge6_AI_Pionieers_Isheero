import os
import json
import torch
import numpy as np
import librosa
import torchaudio
import whisper
from transformers import WhisperProcessor

def load_audio(file_path, sr=16000):
    """
    Charge le fichier audio en le rééchantillonnant à 16 kHz.

    Parameters:
        file_path (str): Chemin du fichier audio.
        sr (int): La fréquence d'échantillonnage cible (16 kHz pour Whisper).

    Returns:
        np.ndarray: Signal audio mono, rééchantillonné à 16 kHz, avec des valeurs typiquement entre -1 et 1.
    """
    # Librosa rééchantillonne l'audio au sample rate spécifié (ici, 16000 Hz)
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio

def load_corpus(corpus_path='dico_corpus.json'):
    """
    Charge le dictionnaire du corpus depuis le fichier JSON.
    
    Parameters:
        corpus_path (str): Chemin vers le fichier JSON du corpus.
    
    Returns:
        dict: Dictionnaire contenant les métadonnées des fichiers audio.
    """
    with open(corpus_path, 'r') as f:
        dico_corpus = json.load(f)
    return dico_corpus

def get_device():
    """
    Détermine le périphérique à utiliser (CUDA ou CPU).
    
    Returns:
        str: 'cuda' si GPU disponible, sinon 'cpu'
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_whisper_model(model_name="tiny", language="fr", task="transcribe"):
    """
    Charge un modèle Whisper pré-entrainé.
    
    Parameters:
        model_name (str): Taille du modèle ('tiny', 'base', 'small', 'medium', 'large').
        language (str): Code de langue pour la transcription.
        task (str): Tâche à effectuer ('transcribe' ou 'translate').
    
    Returns:
        model: Modèle Whisper chargé.
    """
    model = whisper.load_model(model_name)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    return model

def get_whisper_processor(model_name="tiny", language="fr", task="transcribe"):
    """
    Charge un processeur Whisper.
    
    Parameters:
        model_name (str): Taille du modèle ('tiny', 'base', 'small', 'medium', 'large').
        language (str): Code de langue pour la transcription.
        task (str): Tâche à effectuer ('transcribe' ou 'translate').
    
    Returns:
        processor: Processeur Whisper configuré.
    """
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{model_name}",
        language=language,
        task=task
    )
    return processor

def transcribe_audio(model, audio_path, options=None):
    """
    Transcrit un fichier audio avec le modèle Whisper.
    
    Parameters:
        model: Modèle Whisper chargé.
        audio_path (str): Chemin vers le fichier audio.
        options (WhisperDecodingOptions): Options de décodage.
    
    Returns:
        str: Texte transcrit.
    """
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    result = model.decode(mel, options)
    return result.text