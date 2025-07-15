import gradio as gr
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline import VoiceSentimentPipeline
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Initialiser le pipeline
pipeline = VoiceSentimentPipeline()

def process_audio_file(audio_file):
    """
    Traite un fichier audio via l'interface Gradio
    """
    if audio_file is None:
        return "Veuillez uploader un fichier audio", "", "", ""
    
    try:
        result = pipeline.process_audio(audio_file)
        
        if 'error' in result:
            return f"Erreur: {result['error']}", "", "", ""
        
        return (
            result['transcription'],
            result['sentiment'],
            f"{result['confidence']:.2%}",
            f"Durée: {result['audio_duration']:.1f}s | Traitement: {result['processing_time']:.1f}s"
        )
    except Exception as e:
        return f"Erreur inattendue: {str(e)}", "", "", ""

def process_text_only(text):
    """
    Traite uniquement du texte pour l'analyse de sentiment
    """
    if not text or text.strip() == "":
        return "Veuillez saisir du texte", ""
    
    try:
        result = pipeline.process_text_only(text)
        
        if 'error' in result:
            return f"Erreur: {result['error']}", ""
        
        return (
            result['sentiment'],
            f"{result['confidence']:.2%}"
        )
    except Exception as e:
        return f"Erreur inattendue: {str(e)}", ""

# Interface pour l'audio
audio_interface = gr.Interface(
    fn=process_audio_file,
    inputs=gr.Audio(type="filepath", label="Fichier Audio (WAV, MP3, FLAC)"),
    outputs=[
        gr.Textbox(label="Transcription", lines=5),
        gr.Textbox(label="Sentiment Détecté"),
        gr.Textbox(label="Confiance"),
        gr.Textbox(label="Informations")
    ],
    title="🎙️ Analyse Audio → Sentiment",
    description="Uploadez un fichier audio pour obtenir la transcription et l'analyse de sentiment automatique",
    examples=[
        ["data/samples/sample_positive.wav"],
        ["data/samples/sample_negative.wav"]
    ]
)

# Interface pour le texte seul
text_interface = gr.Interface(
    fn=process_text_only,
    inputs=gr.Textbox(label="Texte à analyser", lines=5, placeholder="Saisissez votre texte ici..."),
    outputs=[
        gr.Textbox(label="Sentiment Détecté"),
        gr.Textbox(label="Confiance")
    ],
    title="📝 Analyse Texte → Sentiment",
    description="Analysez directement le sentiment d'un texte",
    examples=[
        ["Je suis très satisfait de ce service, merci beaucoup !"],
        ["Ce produit ne fonctionne pas du tout, je suis très déçu."],
        ["Le service client a été correct, sans plus."]
    ]
)

# Créer l'interface avec onglets
demo = gr.TabbedInterface([audio_interface, text_interface], 
                         ["Analyse Audio", "Analyse Texte"])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
