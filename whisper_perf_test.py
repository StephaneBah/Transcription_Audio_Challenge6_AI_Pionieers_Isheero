import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython.display import display
from jiwer import wer
import whisper

from utils import load_corpus, get_device, get_whisper_model, transcribe_audio

class WhisperTester:
    def __init__(
        self, 
        audio_path="./Collecte_Audio_Challence6_AI_Pionieers_Isheero", 
        model_name="base",
        language="fr",
        output_file="transcription_results.csv"
    ):
        self.audio_path = audio_path
        self.model_name = model_name
        self.language = language
        self.output_file = output_file
        self.device = get_device()
        
        # Charger le modèle
        self.model = get_whisper_model(model_name)
        
        # Options de décodage
        self.options = whisper.DecodingOptions(language=language, without_timestamps=True)
        
        # Charger le corpus
        self.dico_corpus = load_corpus()
        
    def run_transcription(self):
        """Exécute la transcription sur tous les fichiers audio et évalue les performances."""
        # Initialiser une liste pour stocker les résultats
        results = []
        
        # Parcourir les éléments du dictionnaire
        for key, value in tqdm(self.dico_corpus.items(), desc="Transcription des fichiers"):
            identifiant = value["identifiant"]
            contenu_reel = value["contenu"]
            folder = f"{self.audio_path}/{value['identifiant']}"
            
            # Parcourir tous les fichiers audio dans le dossier
            for fname in os.listdir(folder):
                if fname.endswith(('.wav', '.flac', '.mp3', '.webm')):
                    audio_path = os.path.join(folder, fname)
                    
                    # Transcrire l'audio avec Whisper
                    try:
                        contenu_whisper = transcribe_audio(self.model, audio_path, self.options)
                    except Exception as e:
                        print(f"Erreur lors de la transcription de {audio_path}: {e}")
                        contenu_whisper = ""  # En cas d'erreur, on met une chaîne vide
                    
                    # Ajouter les résultats à la liste
                    results.append([identifiant, contenu_reel, contenu_whisper])
        
        # Créer un DataFrame Pandas à partir des résultats
        self.df = pd.DataFrame(results, columns=["Identifiant", "Contenu Réel", "Contenu Whisper"])
        return self.df
    
    def calculate_wer(self):
        """Calcule le Word Error Rate (WER) pour chaque transcription."""
        if not hasattr(self, 'df'):
            print("Aucune transcription trouvée. Exécutez run_transcription() d'abord.")
            return None
        
        # Calculer le WER pour chaque ligne
        wer_scores = []
        for index, row in self.df.iterrows():
            reference = row["Contenu Réel"]
            hypothesis = row["Contenu Whisper"]
            wer_score = wer(reference, hypothesis)
            wer_scores.append(wer_score)
        
        # Ajouter les scores WER comme nouvelle colonne
        self.df['WER'] = wer_scores
        
        # Calculer le WER moyen
        self.average_wer = np.mean(wer_scores)
        print(f"WER moyen: {self.average_wer:.4f}")
        
        return self.df
    
    def save_results(self):
        """Sauvegarde les résultats dans un fichier CSV."""
        if not hasattr(self, 'df'):
            print("Aucun résultat à sauvegarder. Exécutez run_transcription() d'abord.")
            return False
        
        # Sauvegarder le DataFrame dans un fichier CSV
        self.df.to_csv(self.output_file, index=False)
        print(f"Résultats sauvegardés dans {self.output_file}")
        return True


def main():
    # Créer une instance de WhisperTester
    tester = WhisperTester(
        model_name="base",  # Vous pouvez changer en "tiny", "small", "medium", "large"
        output_file="transcription_results2.csv"
    )
    
    # Exécuter les transcriptions
    tester.run_transcription()
    
    # Calculer le WER
    tester.calculate_wer()
    
    # Sauvegarder les résultats
    tester.save_results()


if __name__ == "__main__":
    main()