#!/usr/bin/env python3
"""
Point d'entrée principal pour le projet de transcription audio avec Whisper.
Ce script permet de lancer soit l'entraînement du modèle Whisper (fine-tuning),
soit l'évaluation des performances du modèle.
"""

import argparse
import sys
from whisper_fine_tuning import WhisperFineTuner
from whisper_perf_test import WhisperTester


def run_fine_tuning(args):
    """Lance le processus de fine-tuning du modèle Whisper."""
    print(f"Lancement du fine-tuning avec le modèle '{args.model}'...")
    
    fine_tuner = WhisperFineTuner(
        audio_path=args.audio_path,
        model_name=args.model,
        language=args.language,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub
    )
    
    # Exécuter la chaîne complète d'entraînement
    fine_tuner.prepare_dataset()
    
    if args.freeze_encoder:
        fine_tuner.freeze_encoder()
    
    fine_tuner.create_data_collator()
    fine_tuner.setup_training_args()
    trainer = fine_tuner.train()
    
    print(f"Fine-tuning terminé. Modèle sauvegardé dans {args.output_dir}")
    return trainer


def run_performance_test(args):
    """Lance l'évaluation des performances du modèle Whisper."""
    print(f"Lancement de l'évaluation avec le modèle '{args.model}'...")
    
    tester = WhisperTester(
        audio_path=args.audio_path,
        model_name=args.model,
        language=args.language,
        output_file=args.output_file
    )
    
    # Exécuter la transcription et l'évaluation
    tester.run_transcription()
    tester.calculate_wer()
    tester.save_results()
    
    print(f"Évaluation terminée. Résultats sauvegardés dans {args.output_file}")
    return tester


def main():
    """Point d'entrée principal du programme."""
    # Créer le parser d'arguments principal
    parser = argparse.ArgumentParser(
        description="Projet de transcription audio avec Whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Créer les sous-parseurs pour les différentes commandes
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Sous-parseur pour la commande 'train' (fine-tuning)
    train_parser = subparsers.add_parser("train", help="Fine-tuner le modèle Whisper")
    train_parser.add_argument(
        "--model", 
        type=str, 
        default="tiny", 
        choices=["tiny", "base", "small", "medium", "large"],
        help="Taille du modèle Whisper à utiliser"
    )
    train_parser.add_argument(
        "--audio-path", 
        type=str, 
        default="./Collecte_Audio_Challence6_AI_Pionieers_Isheero",
        help="Chemin vers le dossier contenant les fichiers audio"
    )
    train_parser.add_argument(
        "--language", 
        type=str, 
        default="fr",
        help="Code de langue à utiliser pour la transcription"
    )
    train_parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./whisper-tiny-rad-fr",
        help="Dossier de sortie pour le modèle fine-tuné"
    )
    train_parser.add_argument(
        "--push-to-hub", 
        action="store_true", 
        help="Pousser le modèle sur Hugging Face Hub"
    )
    train_parser.add_argument(
        "--freeze-encoder", 
        action="store_true", 
        help="Geler l'encodeur pendant le fine-tuning"
    )
    
    # Sous-parseur pour la commande 'test' (évaluation des performances)
    test_parser = subparsers.add_parser("test", help="Tester les performances du modèle Whisper")
    test_parser.add_argument(
        "--model", 
        type=str, 
        default="base", 
        choices=["tiny", "base", "small", "medium", "large"],
        help="Taille du modèle Whisper à utiliser"
    )
    test_parser.add_argument(
        "--audio-path", 
        type=str, 
        default="./Collecte_Audio_Challence6_AI_Pionieers_Isheero",
        help="Chemin vers le dossier contenant les fichiers audio"
    )
    test_parser.add_argument(
        "--language", 
        type=str, 
        default="fr",
        help="Code de langue à utiliser pour la transcription"
    )
    test_parser.add_argument(
        "--output-file", 
        type=str, 
        default="transcription_results.csv",
        help="Fichier de sortie pour les résultats de transcription"
    )
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Exécuter la commande appropriée
    if args.command == "train":
        run_fine_tuning(args)
    elif args.command == "test":
        run_performance_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()