import os
import json
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from dotenv import load_dotenv
from huggingface_hub import login
import evaluate

from utils import load_corpus, get_whisper_processor, get_device

class WhisperFineTuner:
    def __init__(
        self, 
        audio_path="./Collecte_Audio_Challence6_AI_Pionieers_Isheero", 
        model_name="tiny",
        language="fr",
        task="transcribe",
        output_dir="./whisper-tiny-rad-fr",
        push_to_hub=False
    ):
        self.audio_path = audio_path
        self.model_name = model_name
        self.language = language
        self.task = task
        self.output_dir = output_dir
        self.push_to_hub = push_to_hub
        
        # Charger HF Token si push_to_hub est True
        if push_to_hub:
            load_dotenv()
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
            else:
                print("Warning: HF_TOKEN not found in .env file. push_to_hub may fail.")
        
        # Charger le processor et le modèle
        self.processor = get_whisper_processor(model_name, language, task)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{model_name}"
        )
        self.model.generation_config.language = language
        self.model.generation_config.task = task
        
        # Initialiser l'évaluateur WER
        self.metric = evaluate.load("wer")
    
    def prepare_dataset(self):
        """Prépare le dataset pour le fine-tuning."""
        # Charger le dictionnaire dico_corpus.json
        dico_corpus = load_corpus()
        
        # Construire la liste des enregistrements
        records = []
        for idx, meta in dico_corpus.items():
            folder = f"{self.audio_path}/{meta['identifiant']}"
            if not os.path.exists(folder):  # Vérification de l'existence du dossier
                print(f"[INFO] Dossier non trouvé : {folder}, ignoré.")
                continue
            
            for fname in os.listdir(folder):
                if fname.endswith(('.wav', '.flac', '.mp3', '.webm')):
                    records.append({
                        "audio": os.path.join(folder, fname),
                        "text": meta['contenu']
                    })
        
        # Créer le dataset à partir de la liste
        ds = Dataset.from_list(records)
        
        # Préparer les données audio et les tokens
        ds_prepared = ds.map(
            self.prepare_batch,
            remove_columns=["audio", "text"],
            num_proc=1
        )
        
        # Enlever les valeurs None du dataset
        ds_prepared = ds_prepared.filter(lambda example: example is not None)
        
        # Split: 90% train, 10% test
        split_dataset = ds_prepared.train_test_split(test_size=0.1, seed=42)
        
        # Organiser sous forme de DatasetDict
        self.ds_prepared = DatasetDict({
            "train": split_dataset["train"],
            "test": split_dataset["test"]
        })
        
        print(f"Dataset préparé: {self.ds_prepared}")
        return self.ds_prepared
    
    def prepare_batch(self, batch):
        """Prépare un batch pour l'entraînement."""
        try:
            waveform, sr = torchaudio.load(batch["audio"])
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            
            # waveform est Tensor [1, T]
            audio = waveform.squeeze().numpy()
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Créer le masque d'attention
            attention_mask = torch.ones_like(inputs.input_features)
            attention_mask = attention_mask.masked_fill(inputs.input_features.eq(0), 0)
            
            # Tokenization des labels
            labels = self.processor.tokenizer(batch["text"], return_tensors="pt").input_ids
            
            return {
                "input_features": inputs.input_features[0],
                "attention_mask": attention_mask[0],
                "labels": labels[0]
            }
        except Exception as e:
            print(f"[ERREUR] Fichier problématique: {batch['audio']}")
            print(f"Exception: {type(e).__name__} — {e}")
            return None
    
    def create_data_collator(self):
        """Crée un collator pour gérer le padding des données."""
        @dataclass
        class DataCollatorSpeechSeq2SeqWithPadding:
            processor: Any
            decoder_start_token_id: int
            max_audio_len: int = 30 * 16000  # 30s at 16kHz
            max_label_len: int = 448
            
            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
                
                label_features = [{"input_ids": feature["labels"]} for feature in features]
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
                
                labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
                
                # Truncate labels to max_label_len
                labels = labels[:, :self.max_label_len]
                
                # Retirer le token de début s'il est déjà là
                if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                    labels = labels[:, 1:]
                
                # Ajouter manuellement le decoder_start_token_id
                decoder_start_tokens = torch.full(
                    (labels.size(0), 1), 
                    self.decoder_start_token_id, 
                    dtype=torch.long
                )
                labels = torch.cat([decoder_start_tokens, labels], dim=1)
                
                batch["labels"] = labels
                return batch
        
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )
        return self.data_collator
    
    def compute_metrics(self, pred):
        """Calcule les métriques (WER) pour l'évaluation."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # we do not want to group tokens when computing the metrics
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    def freeze_encoder(self):
        """Gèle l'encodeur pour le fine-tuning."""
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        print("Encodeur gelé pour le fine-tuning.")
    
    def setup_training_args(self):
        """Configure les arguments d'entraînement."""
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            warmup_steps=500,
            max_steps=5000,
            predict_with_generate=True,
            generation_max_length=225,
            gradient_checkpointing=True,
            learning_rate=3e-5,
            num_train_epochs=3,
            fp16=True,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=self.push_to_hub,
        )
        return self.training_args
    
    def train(self):
        """Entraîne le modèle Whisper."""
        # Vérifier que le dataset est préparé
        if not hasattr(self, 'ds_prepared'):
            self.prepare_dataset()
        
        # Vérifier que les autres composants sont prêts
        if not hasattr(self, 'data_collator'):
            self.create_data_collator()
        
        if not hasattr(self, 'training_args'):
            self.setup_training_args()
        
        # Créer le trainer
        self.trainer = Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=self.ds_prepared["train"],
            eval_dataset=self.ds_prepared["test"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.processor.feature_extractor,
        )
        
        # Lancer l'entraînement
        print("Démarrage de l'entraînement...")
        self.trainer.train()
        
        # Sauvegarder le modèle final
        self.trainer.save_model(self.output_dir)
        print(f"Modèle sauvegardé dans {self.output_dir}")
        
        if self.push_to_hub:
            kwargs = {
                "language": self.language,
                "model_name": f"Whisper {self.model_name.capitalize()} Fr - Radiologie",
                "finetuned_from": f"openai/whisper-{self.model_name}",
                "tasks": "automatic-speech-recognition",
            }
            self.trainer.push_to_hub(**kwargs)
            print("Modèle poussé vers le Hub Hugging Face")
        
        return self.trainer


def main():
    # Créer une instance de WhisperFineTuner
    fine_tuner = WhisperFineTuner(
        model_name="tiny", 
        push_to_hub=True
    )
    
    # Préparer le dataset
    fine_tuner.prepare_dataset()
    
    # Optionnel: geler l'encodeur pour accélérer le fine-tuning
    fine_tuner.freeze_encoder()
    
    # Créer le data collator
    fine_tuner.create_data_collator()
    
    # Configurer les arguments d'entraînement
    fine_tuner.setup_training_args()
    
    # Lancer l'entraînement
    trainer = fine_tuner.train()


if __name__ == "__main__":
    main()