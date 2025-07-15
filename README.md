# Inkriptor API

<img width="1919" height="907" alt="image" src="https://github.com/user-attachments/assets/1998d2a3-849e-482a-ad1b-4be277262f8c" />


A Django REST Framework API providing blog title generation and audio transcription services.

## 🚀 Features

- **Blog Title Generation**: AI-powered title suggestions for blog content
- **Audio Transcription**: Convert audio files to text with MultiLingual Support
- **Modern UI**: Clean interface with API documentation
- **REST API**: Well-documented endpoints with browsable API interface

## 🧩 Feature Workflows

### Blog Title Generation (Local NLP-powered)
1. **User submits blog content** via API (POST JSON).
2. **Text preprocessing**: Lowercase, remove punctuation, tokenize, remove stopwords, and lemmatize using NLTK.
3. **Keyphrase extraction**: Use KeyBERT (local model) to extract top multi-word keyphrases.
4. **Template-based title generation**: Generate titles using smart templates and extracted keyphrases (no first sentence, no repeated keywords).
5. **Filtering**: Remove titles with repeated words/phrases and ensure natural length.
6. **Top suggestions returned** in API response.

   <img width="737" height="714" alt="image" src="https://github.com/user-attachments/assets/eee4cf49-5621-43fd-9ee8-e425eb72e2a6" />


### Audio Transcription with Speaker Diarization
1. **User uploads audio file** via API (POST multipart/form-data).
2. **Audio preprocessing**: Ensure format compatibility (ffmpeg if needed).
3. **Transcription**: Use Whisper model to convert speech to text and detect language.
4. **Speaker diarization**: Segment and cluster speech by speaker using pyannote.audio (NLP-based clustering and feature extraction).
5. **Label formatting**: Assign readable speaker labels and convert language codes to names.
6. **Structured transcript returned** with speaker segments and language info.

<img width="475" height="841" alt="image" src="https://github.com/user-attachments/assets/be2ce354-4ecf-4888-b48a-7148554ff2e6" />


## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Saarcasmic/Inkriptor.git
cd Inkriptor
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
DJANGO_SECRET_KEY=''
HF_AUTH_TOKEN=''
```

5. Run migrations:
```bash
python manage.py migrate
```

6. Start the development server:
```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000`

## 🔌 API Endpoints

### Blog Title Generation

**Endpoint**: `/api/blog/suggest-title/`  
**Method**: POST  
**Content-Type**: application/json

#### Request Format:
```json
{
    "content": "Your blog post content here..."
}
```

#### Sample Response:
```json
{
    "title": "Suggested blog title",
    "alternatives": [
        "Alternative title 1",
        "Alternative title 2"
    ]
}
```

#### cURL Example:
```bash
curl -X POST http://localhost:8000/api/blog/suggest-title/ \
     -H "Content-Type: application/json" \
     -d '{"content": "Your blog post content here..."}'
```

### Audio Transcription

**Endpoint**: `/api/audio/transcribe/`  
**Method**: POST  
**Content-Type**: multipart/form-data

#### Request Format:
- Form data with audio file

#### Sample Response:
```json
{
    "text": "Transcribed text content...",
    "duration": "1:30",
    "language": "en"
}
```

#### cURL Example:
```bash
curl -X POST http://localhost:8000/api/audio/transcribe/ \
     -F "file=@path/to/your/audio.mp3"
```


## 📝 Development Guidelines

- Follow PEP 8 style guide for Python code
- Use descriptive variable names
- Write tests for new features
- Document API changes
- Use meaningful commit messages

## 🔐 Security Considerations

- Keep `SECRET_KEY` secure and unique per deployment
- Use environment variables for sensitive data
- Implement rate limiting for production
- Regular security updates
- Validate file uploads

## 📦 Dependencies

Key packages and their purposes:
- Django REST Framework: API development
- python-dotenv: Environment configuration
- django-cors-headers: CORS support
- Pillow: Image processing
- pydantic: Data validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
