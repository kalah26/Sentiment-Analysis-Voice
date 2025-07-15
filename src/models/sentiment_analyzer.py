from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from typing import Dict, List
import logging

class SentimentAnalyzer:
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initialise l'analyseur de sentiment avec BERT
        
        Args:
            model_name: Nom du modèle HuggingFace à utiliser
        """
        self.model_name = model_name
        
        # Vérifier si GPU est disponible
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Initialiser le pipeline
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device
        )
        
        logging.info(f"Analyseur de sentiment initialisé sur {'GPU' if self.device >= 0 else 'CPU'}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Préprocesse le texte pour l'analyse de sentiment
        
        Args:
            text: Texte à préprocesser
            
        Returns:
            Texte préprocessé
        """
        # Supprimer les caractères spéciaux excessifs
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Nettoyer et normaliser
        text = text.strip().lower()
        
        return text
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyse le sentiment d'un texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire contenant les résultats d'analyse
        """
        try:
            if not text or text.strip() == "":
                return {
                    "sentiment": "Neutre",
                    "confidence": 0.0,
                    "error": "Texte vide"
                }
            
            # Préprocesser le texte
            processed_text = self.preprocess_text(text)
            
            # Diviser le texte en chunks si trop long
            max_length = 512
            chunks = self._split_text(processed_text, max_length)
            
            # Analyser chaque chunk
            results = []
            for chunk in chunks:
                result = self.classifier(chunk)
                results.append(result[0])
            
            # Agréger les résultats
            aggregated_result = self._aggregate_results(results)
            
            # Mapper vers nos classes
            sentiment_mapping = self._map_sentiment(aggregated_result['label'])
            
            return {
                "sentiment": sentiment_mapping,
                "confidence": aggregated_result['score'],
                "raw_results": results,
                "processed_text": processed_text,
                "model_used": self.model_name
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de sentiment: {str(e)}")
            return {"error": str(e)}
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Divise le texte en chunks de taille maximale"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Agrège les résultats de plusieurs chunks"""
        if len(results) == 1:
            return results[0]
        
        # Calculer la moyenne pondérée
        positive_score = sum(r['score'] for r in results if r['label'] == 'POSITIVE')
        negative_score = sum(r['score'] for r in results if r['label'] == 'NEGATIVE')
        
        if positive_score > negative_score:
            return {'label': 'POSITIVE', 'score': positive_score / len(results)}
        else:
            return {'label': 'NEGATIVE', 'score': negative_score / len(results)}
    
    def _map_sentiment(self, label: str) -> str:
        """Mappe les labels du modèle vers nos classes"""
        mapping = {
            'POSITIVE': 'Satisfait',
            'NEGATIVE': 'Mécontent',
            'NEUTRAL': 'Neutre'
        }
        return mapping.get(label, 'Neutre')