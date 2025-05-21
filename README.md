# 🎙️ Transcription Audio avec Whisper - Challenge 6 AI Pioneers

<div align="center">
  
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Status](https://img.shields.io/badge/Status-Active-success)

</div>

**[🔗 Accéder à l'application de collecte d'audio](https://collecte-audio-production-3da5.up.railway.app/)**

**[📖 Présentation de l'architecture Whisper](https://presentation-de-larchite-e93v5v7.gamma.site/)**

**[📂 Docs sur Whisper](https://en.wikipedia.org/wiki/Whisper_%28speech_recognition_system%29)**


## 📋 Vue d'ensemble

Ce projet utilise le modèle **Whisper** pour effectuer la transcription automatique de fichiers audio en texte. Il comprend deux tâches principales : 
1. **🔄 Fine-tuning du modèle Whisper** sur un corpus spécifique.
2. **📊 Évaluation des performances** du modèle sur des données de test.

## 📂 Structure du projet

| Fichier | Description |
|---------|-------------|
| **`main.py`** | Point d'entrée principal pour lancer le fine-tuning ou l'évaluation des performances |
| **`whisper_fine_tuning.py`** | Script pour le fine-tuning du modèle Whisper |
| **`whisper_perf_test.py`** | Script pour tester les performances du modèle Whisper |
| **`utils.py`** | Fonctions utilitaires pour la manipulation des données audio et du modèle |
| **`requirements.txt`** | Liste des dépendances nécessaires |
| **`dico_corpus.json`** | Fichier contenant les métadonnées des fichiers audio et leurs transcriptions |

---

## ⚙️ Prérequis

<details open>
<summary><b>🛠️ Environnement requis</b></summary>
<br>

1. **🐍 Python** : Assurez-vous d'avoir Python 3.10 ou une version ultérieure installée <=3.12
2. **🖥️ CUDA** : Si vous souhaitez utiliser un GPU, installez CUDA et configurez PyTorch pour l'utiliser.
3. **🎬 FFmpeg** : Installez FFmpeg pour le traitement des fichiers audio :
   ```bash
   sudo apt update && sudo apt install ffmpeg
   ```
</details>

## 🚀 Installation

<details open>
<summary><b>Étapes d'installation</b></summary>
<br>

### Etape 1: Cloner le dépôt
```bash
git clone <URL_DU_DEPOT>
cd Transcription_Audio_Challence6_AI_Pionieers_Isheero
```

### Etape 2: Créer un environnement virtuel
```bash
sudo apt install python3.12-venv #(optionnel et dépend de votre version de python)
python3 -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

### Etape 3: Installer les dépendances

#### Prérequis pour les dépendances
```bash
# Installer cURL & tar & autres (optionnel)
sudo apt update && sudo apt install curl tar 
sudo apt install -y pkg-config libssl-dev build-essential


# Installer Rust (nécessaire pour tokenizers) (optionnel)
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"

# Installer d'autres dépendances de compilation
sudo apt update && sudo apt install -y build-essential python3-dev
```

#### Option A: Installation avec pip
```bash
pip install --upgrade setuptools wheel pip
pip install numpy==2.2.0
pip install -r requirements.txt
pip install git+https://github.com/openai/whisper.git
```

#### Option B: Installation avec UV (plus rapide)
```bash
# Installer UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances avec UV
uv venv venv  # Créer un environnement virtuel (si ce n'est pas déjà fait)
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
uv pip install --upgrade setuptools wheel
uv pip install numpy==2.2.0
uv pip install -r requirements.txt
uv pip install git+https://github.com/openai/whisper.git
```
</details>

### Etape 4: Configuration de l'API HuggingFace

Créez un compte sur [HuggingFace](https://huggingface.co)
Générez un jeton d'API [ici](https://huggingface.co/settings/tokens).
Insérez dans le fichier `.env` la ligne suivante :
```properties
HF_TOKEN = votre_jeton_huggingface
```

## 📝 Utilisation

### 🔄 Fine-tuning du modèle Whisper

Pour lancer le fine-tuning du modèle Whisper, utilisez la commande suivante:

```bash
python main.py train --model tiny --output-dir ./whisper-tiny-rad-fr
```

<details>
<summary><b>Options disponibles</b></summary>
<br>

| Option | Description | Défaut |
|--------|-------------|--------|
| `--model` | Taille du modèle à utiliser ('tiny', 'base', 'small', 'medium', 'large') | 'tiny' |
| `--audio-path` | Chemin vers le dossier contenant les fichiers audio | './Collecte_Audio_Challence6_AI_Pionieers_Isheero' |
| `--language` | Code de langue à utiliser | 'fr' |
| `--output-dir` | Dossier où sauvegarder le modèle fine-tuné | './whisper-tiny-rad-fr' |
| `--push-to-hub` | Ajouter ce flag pour pousser le modèle sur Hugging Face Hub | - |
| `--freeze-encoder` | Ajouter ce flag pour geler l'encodeur pendant le fine-tuning | - |

</details>

**Exemple avec toutes les options:**
```bash
python main.py train --model small --audio-path ./mon_dossier_audio --language fr --output-dir ./mon_modele_fine_tune --push-to-hub --freeze-encoder
```

### 📊 Test de performances du modèle

Pour évaluer les performances du modèle Whisper (pré-entrainé ou fine-tuné), utilisez:

```bash
python main.py test --model base --output-file transcription_results.csv
```
NB: Pour le moment, les commandes sont configurés pour tester que les modèles pré-entraînés de Whisper.

<details>
<summary><b>Options disponibles</b></summary>
<br>

| Option | Description | Défaut |
|--------|-------------|--------|
| `--model` | Taille du modèle à utiliser ('tiny', 'base', 'small', 'medium', 'large') | 'base' |
| `--audio-path` | Chemin vers le dossier contenant les fichiers audio | './Collecte_Audio_Challence6_AI_Pionieers_Isheero' |
| `--language` | Code de langue à utiliser | 'fr' |
| `--output-file` | Fichier CSV où sauvegarder les résultats | 'transcription_results.csv' |

</details>

**Exemple avec toutes les options:**
```bash
python main.py test --model large --audio-path ./mon_dossier_audio --language fr --output-file mes_resultats.csv
```

## 🧠 Utilisation d'un modèle fine-tuné

Pour utiliser un modèle que vous avez fine-tuné (stocké localement), vous pouvez modifier le code dans `whisper_perf_test.py` pour charger le modèle depuis le dossier où il a été sauvegardé.

## 📈 Résultats

- Les résultats des transcriptions et les scores WER (Word Error Rate) sont sauvegardés dans un fichier CSV (défini par l'option `--output-file`).
- Les métriques d'entraînement et de validation sont suivies avec TensorBoard.

Pour visualiser les métriques avec TensorBoard :
```bash
tensorboard --logdir=./whisper-tiny-rad-fr
```

---

<div align="center">
  
Développé dans le cadre du **Challenge 6 AI Pioneers - Isheero** 🚀

</div>

