from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import sys
import logging
from typing import Optional

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline import VoiceSentimentPipeline

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Initialiser l'application FastAPI
app = FastAPI(
    title="Voice Sentiment Analysis API",
    description="API pour l'analyse de sentiment dans les appels vocaux",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser le pipeline
pipeline = VoiceSentimentPipeline()

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyse un fichier audio et retourne la transcription + sentiment
    """
    # Vérifier le format du fichier
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        raise HTTPException(
            status_code=400, 
            detail="Format audio non supporté. Utilisez WAV, MP3, FLAC ou M4A"
        )
    
    # Vérifier la taille du fichier (max 50MB)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(
            status_code=400,
            detail="Fichier trop volumineux. Maximum 50MB"
        )
    
    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Traiter le fichier
        result = pipeline.process_audio(tmp_file_path)
        
        # Ajouter des métadonnées
        result['file_info'] = {
            'filename': file.filename,
            'size_mb': file_size / (1024 * 1024),
            'content_type': file.content_type
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Erreur lors du traitement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Supprimer le fichier temporaire
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/analyze-text")
async def analyze_text(text: str = Form(...)):
    """
    Analyse le sentiment d'un texte
    """
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Texte vide")
    
    if len(text) > 10000:  # Limite de 10k caractères
        raise HTTPException(status_code=400, detail="Texte trop long (max 10000 caractères)")
    
    try:
        result = pipeline.process_text_only(text)
        
        # Ajouter des métadonnées
        result['text_info'] = {
            'length': len(text),
            'word_count': len(text.split())
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Vérification de l'état de l'API
    """
    return {
        "status": "healthy",
        "models_loaded": True,
        "timestamp": pipeline.process_text_only("test")["timestamp"]
    }

@app.get("/models-info")
async def models_info():
    """
    Informations sur les modèles utilisés
    """
    return {
        "speech_to_text": {
            "model": "facebook/wav2vec2-large-960h-lv60-self",
            "description": "Wav2Vec2 pour la transcription audio"
        },
        "sentiment_analysis": {
            "model": "nlptown/bert-base-multilingual-uncased-sentiment",
            "description": "BERT multilingue pour l'analyse de sentiment"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
