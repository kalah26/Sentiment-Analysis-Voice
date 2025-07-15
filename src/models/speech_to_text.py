# import torch
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
# import librosa
# import numpy as np
# from typing import Optional, Union
# import logging

# class SpeechToText:
#     def __init__(self, model_name: str = "facebook/wav2vec2-large-960h-lv60-self"):
#         """
#         Initialise le modèle Speech-to-Text avec Wav2Vec2
        
#         Args:
#             model_name: Nom du modèle HuggingFace à utiliser
#         """
#         self.model_name = model_name
#         self.processor = Wav2Vec2Processor.from_pretrained(model_name)
#         self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
#         # Vérifier si GPU est disponible
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        
#         logging.info(f"Modèle Speech-to-Text initialisé sur {self.device}")
    
#     def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
#         """
#         Préprocesse le fichier audio pour la transcription
        
#         Args:
#             audio_path: Chemin vers le fichier audio
#             target_sr: Fréquence d'échantillonnage cible
            
#         Returns:
#             Audio préprocessé sous forme de numpy array
#         """
#         try:
#             # Charger l'audio avec librosa
#             speech, sr = librosa.load(audio_path, sr=target_sr)
            
#             # Normaliser l'audio
#             speech = librosa.util.normalize(speech)
            
#             return speech
#         except Exception as e:
#             logging.error(f"Erreur lors du préprocessing audio: {str(e)}")
#             raise
    
#     def transcribe(self, audio_path: str) -> dict:
#         """
#         Transcrit un fichier audio en texte
        
#         Args:
#             audio_path: Chemin vers le fichier audio
            
#         Returns:
#             Dictionnaire contenant la transcription et les métadonnées
#         """
#         try:
#             # Préprocesser l'audio
#             speech = self.preprocess_audio(audio_path)
            
#             # Tokeniser l'audio
#             inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
#             input_values = inputs.input_values.to(self.device)
            
#             # Inférence
#             with torch.no_grad():
#                 logits = self.model(input_values).logits
            
#             # Décodage
#             predicted_ids = torch.argmax(logits, dim=-1)
#             transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
            
#             # Calculer la confiance moyenne
#             probabilities = torch.softmax(logits, dim=-1)
#             confidence = torch.max(probabilities, dim=-1)[0].mean().item()
            
#             return {
#                 "transcription": transcription.strip(),
#                 "confidence": confidence,
#                 "audio_duration": len(speech) / 16000,
#                 "model_used": self.model_name
#             }
            
#         except Exception as e:
#             logging.error(f"Erreur lors de la transcription: {str(e)}")
#             return {"error": str(e)}

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
import librosa
import numpy as np
from typing import Optional, Union, Literal
import logging

class MultilingualSpeechToText:
    def __init__(self):
        """
        Initialise le modèle Speech-to-Text multilingue avec support pour l'anglais et le français
        """
        # Configuration des modèles
        self.models_config = {
            "english": "facebook/wav2vec2-large-960h-lv60-self",
            "french": "jonatasgrosman/wav2vec2-large-xlsr-53-french"
        }
        
        # Stockage des modèles et processeurs
        self.models = {}
        self.processors = {}
        
        # Vérifier si GPU est disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialiser les modèles
        self._load_models()
        
        logging.info(f"Modèles Speech-to-Text initialisés sur {self.device}")
    
    def _load_models(self):
        """
        Charge les modèles anglais et français
        """
        for language, model_name in self.models_config.items():
            try:
                logging.info(f"Chargement du modèle {language}: {model_name}")
                
                # Charger le processeur et le modèle
                self.processors[language] = Wav2Vec2Processor.from_pretrained(model_name)
                self.models[language] = Wav2Vec2ForCTC.from_pretrained(model_name)
                
                # Déplacer le modèle sur le device approprié
                self.models[language].to(self.device)
                
                logging.info(f"Modèle {language} chargé avec succès")
                
            except Exception as e:
                logging.error(f"Erreur lors du chargement du modèle {language}: {str(e)}")
                raise
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Préprocesse le fichier audio pour la transcription
        
        Args:
            audio_path: Chemin vers le fichier audio
            target_sr: Fréquence d'échantillonnage cible
            
        Returns:
            Audio préprocessé sous forme de numpy array
        """
        try:
            # Charger l'audio avec librosa
            speech, sr = librosa.load(audio_path, sr=target_sr)
            
            # Normaliser l'audio
            speech = librosa.util.normalize(speech)
            
            return speech
        except Exception as e:
            logging.error(f"Erreur lors du préprocessing audio: {str(e)}")
            raise
    
    def _transcribe_with_model(self, speech: np.ndarray, language: str) -> dict:
        """
        Transcrit l'audio avec un modèle spécifique
        
        Args:
            speech: Audio préprocessé
            language: Langue du modèle à utiliser ("english" ou "french")
            
        Returns:
            Résultat de la transcription
        """
        try:
            processor = self.processors[language]
            model = self.models[language]
            
            # Tokeniser l'audio
            inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)
            
            # Inférence
            with torch.no_grad():
                logits = model(input_values).logits
            
            # Décodage
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Calculer la confiance moyenne
            probabilities = torch.softmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0].mean().item()
            
            return {
                "transcription": transcription.strip(),
                "confidence": confidence,
                "language": language,
                "model_used": self.models_config[language]
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de la transcription avec le modèle {language}: {str(e)}")
            return {"error": str(e), "language": language}
    
    def transcribe(self, audio_path: str, language: Optional[Literal["english", "french"]] = None) -> dict:
        """
        Transcrit un fichier audio en texte
        
        Args:
            audio_path: Chemin vers le fichier audio
            language: Langue spécifique à utiliser ("english" ou "french"). 
                     Si None, teste les deux modèles et retourne le meilleur résultat.
            
        Returns:
            Dictionnaire contenant la transcription et les métadonnées
        """
        try:
            # Préprocesser l'audio
            speech = self.preprocess_audio(audio_path)
            audio_duration = len(speech) / 16000
            
            if language:
                # Utiliser le modèle spécifique demandé
                if language not in self.models_config:
                    raise ValueError(f"Langue non supportée: {language}. Langues disponibles: {list(self.models_config.keys())}")
                
                result = self._transcribe_with_model(speech, language)
                result["audio_duration"] = audio_duration
                return result
            
            else:
                # Tester les deux modèles et retourner le meilleur résultat
                results = {}
                
                for lang in self.models_config.keys():
                    result = self._transcribe_with_model(speech, lang)
                    if "error" not in result:
                        results[lang] = result
                
                if not results:
                    return {"error": "Aucun modèle n'a pu traiter l'audio"}
                
                # Sélectionner le résultat avec la meilleure confiance
                best_language = max(results.keys(), key=lambda k: results[k]["confidence"])
                best_result = results[best_language]
                
                # Ajouter les informations sur tous les résultats
                best_result["audio_duration"] = audio_duration
                best_result["all_results"] = results
                best_result["auto_detected_language"] = best_language
                
                return best_result
                
        except Exception as e:
            logging.error(f"Erreur lors de la transcription: {str(e)}")
            return {"error": str(e)}
    
    def transcribe_english(self, audio_path: str) -> dict:
        """
        Transcrit un fichier audio en utilisant spécifiquement le modèle anglais
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Dictionnaire contenant la transcription et les métadonnées
        """
        return self.transcribe(audio_path, language="english")
    
    def transcribe_french(self, audio_path: str) -> dict:
        """
        Transcrit un fichier audio en utilisant spécifiquement le modèle français
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Dictionnaire contenant la transcription et les métadonnées
        """
        return self.transcribe(audio_path, language="french")
    
    def get_available_languages(self) -> list:
        """
        Retourne la liste des langues disponibles
        
        Returns:
            Liste des langues supportées
        """
        return list(self.models_config.keys())
    
    def get_model_info(self) -> dict:
        """
        Retourne les informations sur les modèles chargés
        
        Returns:
            Dictionnaire avec les informations des modèles
        """
        return {
            "device": str(self.device),
            "models": self.models_config,
            "languages_available": self.get_available_languages()
        }

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le modèle multilingue
    stt = MultilingualSpeechToText()
    
    # Exemple 1: Transcription automatique (teste les deux modèles)
    result = stt.transcribe("audio_file.wav")
    print("Transcription automatique:", result)
    
    # Exemple 2: Transcription en anglais spécifiquement
    result_en = stt.transcribe_english("english_audio.wav")
    print("Transcription anglaise:", result_en)
    
    # Exemple 3: Transcription en français spécifiquement
    result_fr = stt.transcribe_french("french_audio.wav")
    print("Transcription française:", result_fr)
    
    # Exemple 4: Obtenir les informations sur les modèles
    info = stt.get_model_info()
    print("Informations des modèles:", info)