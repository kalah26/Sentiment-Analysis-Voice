# Sentiment Analysis Voice API

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)

## Project Description

The **Sentiment Analysis Voice API** is an innovative project that leverages voice input to analyze sentiment in text. This application allows users to speak their thoughts, which are then converted to text and analyzed for sentiment. The project integrates speech-to-text capabilities with sentiment analysis, providing a seamless user experience for understanding emotional tone in spoken language.

### Key Features
- **Speech-to-Text Conversion**: Convert spoken language into text using advanced speech recognition techniques.
- **Sentiment Analysis**: Analyze the emotional tone of the transcribed text to determine sentiment polarity (positive, negative, neutral).
- **API Integration**: Easily deployable as an API for integration with other applications or services.

## Tech Stack

| Technology         | Description                          |
|--------------------|--------------------------------------|
| ![Python](https://img.shields.io/badge/Python-3.10-blue.svg) | Programming language used for backend development. |

## Installation Instructions

### Prerequisites
- Python 3.10 or higher

### Step-by-Step Installation Guide
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mvhamad/Sentiment-Analysis-Voice-APII.git
   cd Sentiment-Analysis-Voice-APII
   ```

2. **Install dependencies**:
   - Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (if applicable):
   - Create a `.env` file in the root directory and define any necessary environment variables. (Refer to the documentation of any specific libraries used for details on required variables.)


## Usage

To run the project, execute the following command to start the API:
```bash
python src/api/main.py
```

### Basic Usage Example
Once the API is running, you can send a POST request to the endpoint with audio data for sentiment analysis. Ensure you follow the API documentation for the required request format.

## Project Structure

The project is organized as follows:

```
Sentiment-Analysis-Voice-APII/
├── src/
│   ├── api/                      # Contains the main API entry point
│   │   └── main.py               # Main application file for the API
│   ├── interface/                # User interface components
│   │   └── gradio_app.py         # Gradio app for user interaction
│   └── models/                   # Contains model definitions and logic
│       ├── pipeline.py           # Pipeline for processing audio input
│       ├── sentiment_analyzer.py  # Logic for analyzing sentiment
│       └── speech_to_text.py     # Logic for converting speech to text
├── .gitignore                    # Specifies files to ignore in version control
├── docker-compose.yml            # Docker Compose configuration file
└── requirements.txt              # List of dependencies
```

### Explanation of Main Directories and Files
- **api/**: This directory contains the main API code, specifically `main.py`, which serves as the entry point for the application.
- **interface/**: This directory includes the user interface components, such as `gradio_app.py`, which allows users to interact with the sentiment analysis functionalities.
- **models/**: Contains the core logic for the application, including:
  - `pipeline.py`: Manages the workflow for processing audio input.
  - `sentiment_analyzer.py`: Implements the sentiment analysis algorithms.
  - `speech_to_text.py`: Handles the conversion of speech to text.

## Contributing

We welcome contributions to enhance the functionality and performance of the Sentiment Analysis Voice API. If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch and submit a pull request.

Thank you for your interest in contributing to this project!
