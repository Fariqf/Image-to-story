# Image-to-story
This repository contains a simple Streamlit web application that takes an uploaded image as input, converts it to text using a pre-trained image-to-text model, generates a short story based on the extracted text, and finally converts the generated story into audio using text-to-speech.

# Getting Started
Prerequisites
Python 3.7 or later
Streamlit
transformers
langchain
requests
dotenv

# Installation
1. Clone this repository:
git clone https://github.com/Fariqf/Image-to-story.git
cd Image-to-story

2. Install the required packages:
   pip install -r requirements.txt

3. Create a .env file in the root directory with the following content:
   OPENAI_API_KEY=your_openai_api_key
    HUGGINGFACEHUB_API_TOKEN=your_huggingfacehub_api_token
4. Run the Streamlit app:
   streamlit run app.py

# How to Use
Upload an image using the file uploader in the sidebar.
The application will display the uploaded image.
Click the "Generate Story" button.
The generated story will be played as audio.


# Components
## Image to Text
The application uses the Salesforce/blip-image-captioning-base model from the Hugging Face Transformers library to convert the uploaded image to text.

## Story Generation
The langchain library is employed to generate a short story based on the extracted text. The OpenAI GPT-3.5-turbo model is used for story generation.

## Text to Speech
The generated story is converted into audio using the ESPnet Kan-bayashi LJ Speech VITS model, accessible through the Hugging Face Inference API.
