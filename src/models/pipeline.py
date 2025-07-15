# from .speech_to_text import SpeechToText
# from .sentiment_analyzer import SentimentAnalyzer
# from datetime import datetime
# import logging
# from typing import Dict, Optional

# class VoiceSentimentPipeline:
#     def __init__(self, 
#                  stt_model: str = "facebook/wav2vec2-large-960h-lv60-self",
#                  sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
#         """
#         Initialise le pipeline complet de traitement
        
#         Args:
#             stt_model: Modèle Speech-to-Text
#             sentiment_model: Modèle d'analyse de sentiment
#         """
#         self.speech_to_text = SpeechToText(stt_model)
#         self.sentiment_analyzer = SentimentAnalyzer(sentiment_model)
        
#         logging.info("Pipeline Voice Sentiment initialisé avec succès")
    
#     def process_audio(self, audio_path: str) -> Dict:
#         """
#         Traite un fichier audio complet: transcription + analyse de sentiment
        
#         Args:
#             audio_path: Chemin vers le fichier audio
            
#         Returns:
#             Dictionnaire contenant tous les résultats
#         """
#         try:
#             start_time = datetime.now()
            
#             # Étape 1: Transcription Speech-to-Text
#             logging.info("Début de la transcription...")
#             transcription_result = self.speech_to_text.transcribe(audio_path)
            
#             if "error" in transcription_result:
#                 return {
#                     "error": f"Erreur transcription: {transcription_result['error']}",
#                     "timestamp": start_time.isoformat()
#                 }
            
#             # Étape 2: Analyse de sentiment
#             logging.info("Début de l'analyse de sentiment...")
#             sentiment_result = self.sentiment_analyzer.analyze_sentiment(
#                 transcription_result['transcription']
#             )
            
#             if "error" in sentiment_result:
#                 return {
#                     "transcription": transcription_result['transcription'],
#                     "error": f"Erreur sentiment: {sentiment_result['error']}",
#                     "timestamp": start_time.isoformat()
#                 }
            
#             # Calculer le temps de traitement
#             processing_time = (datetime.now() - start_time).total_seconds()
            
#             # Résultat final
#             result = {
#                 "transcription": transcription_result['transcription'],
#                 "sentiment": sentiment_result['sentiment'],
#                 "confidence": sentiment_result['confidence'],
#                 "transcription_confidence": transcription_result['confidence'],
#                 "audio_duration": transcription_result['audio_duration'],
#                 "processing_time": processing_time,
#                 "timestamp": start_time.isoformat(),
#                 "models_used": {
#                     "speech_to_text": transcription_result['model_used'],
#                     "sentiment_analysis": sentiment_result['model_used']
#                 }
#             }
            
#             logging.info(f"Traitement terminé en {processing_time:.2f}s")
#             return result
            
#         except Exception as e:
#             logging.error(f"Erreur dans le pipeline: {str(e)}")
#             return {
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }
    
#     def process_text_only(self, text: str) -> Dict:
#         """
#         Traite uniquement du texte pour l'analyse de sentiment
        
#         Args:
#             text: Texte à analyser
            
#         Returns:
#             Résultat de l'analyse de sentiment
#         """
#         try:
#             result = self.sentiment_analyzer.analyze_sentiment(text)
#             result['timestamp'] = datetime.now().isoformat()
#             return result
#         except Exception as e:
#             return {
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }


from .speech_to_text import MultilingualSpeechToText  # Updated import
from .sentiment_analyzer import SentimentAnalyzer
from datetime import datetime
import logging
from typing import Dict, Optional, Literal

class VoiceSentimentPipeline:
    def __init__(self, 
                 sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initialise le pipeline complet de traitement avec support multilingue
        
        Args:
            sentiment_model: Modèle d'analyse de sentiment
        """
        # Initialiser le modèle Speech-to-Text multilingue
        self.speech_to_text = MultilingualSpeechToText()
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_model)
        
        logging.info("Pipeline Voice Sentiment multilingue initialisé avec succès")
    
    def process_audio(self, 
                     audio_path: str, 
                     language: Optional[Literal["english", "french"]] = None) -> Dict:
        """
        Traite un fichier audio complet: transcription + analyse de sentiment
        
        Args:
            audio_path: Chemin vers le fichier audio
            language: Langue spécifique à utiliser pour la transcription 
                     ("english", "french" ou None pour auto-détection)
            
        Returns:
            Dictionnaire contenant tous les résultats
        """
        try:
            start_time = datetime.now()
            
            # Étape 1: Transcription Speech-to-Text
            logging.info(f"Début de la transcription{f' en {language}' if language else ' (auto-détection)'}...")
            transcription_result = self.speech_to_text.transcribe(audio_path, language)
            
            if "error" in transcription_result:
                return {
                    "error": f"Erreur transcription: {transcription_result['error']}",
                    "timestamp": start_time.isoformat()
                }
            
            # Étape 2: Analyse de sentiment
            logging.info("Début de l'analyse de sentiment...")
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                transcription_result['transcription']
            )
            
            if "error" in sentiment_result:
                return {
                    "transcription": transcription_result['transcription'],
                    "language_detected": transcription_result.get('language', 'unknown'),
                    "error": f"Erreur sentiment: {sentiment_result['error']}",
                    "timestamp": start_time.isoformat()
                }
            
            # Calculer le temps de traitement
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Résultat final avec informations multilingues
            result = {
                "transcription": transcription_result['transcription'],
                "sentiment": sentiment_result['sentiment'],
                "confidence": sentiment_result['confidence'],
                "transcription_confidence": transcription_result['confidence'],
                "audio_duration": transcription_result['audio_duration'],
                "processing_time": processing_time,
                "timestamp": start_time.isoformat(),
                "language_used": transcription_result.get('language', 'unknown'),
                "models_used": {
                    "speech_to_text": transcription_result['model_used'],
                    "sentiment_analysis": sentiment_result['model_used']
                }
            }
            
            # Ajouter les informations sur l'auto-détection si applicable
            if 'auto_detected_language' in transcription_result:
                result['auto_detected_language'] = transcription_result['auto_detected_language']
                result['all_transcription_results'] = transcription_result['all_results']
            
            logging.info(f"Traitement terminé en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logging.error(f"Erreur dans le pipeline: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_audio_english(self, audio_path: str) -> Dict:
        """
        Traite un fichier audio en utilisant spécifiquement le modèle anglais
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Dictionnaire contenant tous les résultats
        """
        return self.process_audio(audio_path, language="english")
    
    def process_audio_french(self, audio_path: str) -> Dict:
        """
        Traite un fichier audio en utilisant spécifiquement le modèle français
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Dictionnaire contenant tous les résultats
        """
        return self.process_audio(audio_path, language="french")
    
    def process_audio_with_comparison(self, audio_path: str) -> Dict:
        """
        Traite un fichier audio avec les deux modèles et compare les résultats
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Dictionnaire contenant les résultats des deux modèles
        """
        try:
            start_time = datetime.now()
            
            # Traitement avec auto-détection (qui teste les deux modèles)
            logging.info("Début de la transcription comparative...")
            transcription_result = self.speech_to_text.transcribe(audio_path)
            
            if "error" in transcription_result:
                return {
                    "error": f"Erreur transcription: {transcription_result['error']}",
                    "timestamp": start_time.isoformat()
                }
            
            # Analyser le sentiment pour chaque résultat de transcription
            comparison_results = {}
            
            if 'all_results' in transcription_result:
                for lang, trans_result in transcription_result['all_results'].items():
                    sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                        trans_result['transcription']
                    )
                    
                    comparison_results[lang] = {
                        "transcription": trans_result['transcription'],
                        "transcription_confidence": trans_result['confidence'],
                        "sentiment": sentiment_result.get('sentiment', 'unknown'),
                        "sentiment_confidence": sentiment_result.get('confidence', 0.0),
                        "model_used": trans_result['model_used']
                    }
            
            # Calculer le temps de traitement
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Résultat final avec comparaison
            result = {
                "best_result": {
                    "transcription": transcription_result['transcription'],
                    "language": transcription_result.get('auto_detected_language', 'unknown'),
                    "transcription_confidence": transcription_result['confidence']
                },
                "comparison_results": comparison_results,
                "audio_duration": transcription_result['audio_duration'],
                "processing_time": processing_time,
                "timestamp": start_time.isoformat()
            }
            
            logging.info(f"Traitement comparatif terminé en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logging.error(f"Erreur dans le pipeline comparatif: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_text_only(self, text: str) -> Dict:
        """
        Traite uniquement du texte pour l'analyse de sentiment
        
        Args:
            text: Texte à analyser
            
        Returns:
            Résultat de l'analyse de sentiment
        """
        try:
            result = self.sentiment_analyzer.analyze_sentiment(text)
            result['timestamp'] = datetime.now().isoformat()
            return result
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_pipeline_info(self) -> Dict:
        """
        Retourne les informations sur le pipeline
        
        Returns:
            Dictionnaire avec les informations du pipeline
        """
        return {
            "speech_to_text": self.speech_to_text.get_model_info(),
            "sentiment_analyzer": {
                "model_used": getattr(self.sentiment_analyzer, 'model_name', 'unknown')
            },
            "available_languages": self.speech_to_text.get_available_languages()
        }

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le pipeline
    pipeline = VoiceSentimentPipeline()
    
    # Exemple 1: Traitement avec auto-détection
    result = pipeline.process_audio("audio_file.wav")
    print("Résultat auto-détection:", result)
    
    # Exemple 2: Traitement en anglais spécifiquement
    result_en = pipeline.process_audio_english("english_audio.wav")
    print("Résultat anglais:", result_en)
    
    # Exemple 3: Traitement en français spécifiquement
    result_fr = pipeline.process_audio_french("french_audio.wav")
    print("Résultat français:", result_fr)
    
    # Exemple 4: Comparaison des deux modèles
    comparison = pipeline.process_audio_with_comparison("audio_file.wav")
    print("Comparaison:", comparison)
    
    # Exemple 5: Traitement de texte uniquement
    text_result = pipeline.process_text_only("This is a great product!")
    print("Analyse de texte:", text_result)
    
    # Exemple 6: Informations du pipeline
    info = pipeline.get_pipeline_info()
    print("Informations du pipeline:", info)