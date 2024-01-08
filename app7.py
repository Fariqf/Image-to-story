import streamlit as st
import requests
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os


load_dotenv(find_dotenv())
# HUGGINGFACE_HUB_API=os.getenv("HUGGINGFACE_HUB_API")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# dotenv.load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# image to text
def imgToText(url):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img_to_text(url)[0]['generated_text']
    return text

# LLM
def generate_story(scenario):
    template = """
               You are a story teller.
               You can generate a short story based on a simple narrative, the story should be no more than 40 words:

               CONTEXT: {scenario}
               STORY:
            """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo"), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    return story

def textToSpeech(story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer " + HUGGINGFACEHUB_API_TOKEN}
    payload = {"inputs": story}
    response = requests.post(API_URL, headers=headers, json=payload)
    with open("story.flac", "wb") as f:
        f.write(response.content)

def generate_story_and_play_audio(image):
    scenario = imgToText(image.name)
    story = generate_story(scenario)
    textToSpeech(story)
    return "story.flac"

st.title("Generate Story from Image")
st.sidebar.markdown("### Upload an image:")
uploaded_image = st.sidebar.file_uploader(label="Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.audio(generate_story_and_play_audio(uploaded_image), format="audio/flac")
    st.write()
    st.write()










# import gradio as gr
# import dotenv
# from transformers import pipeline
# from langchain import LLMChain
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# import requests
# import os

# dotenv.load_dotenv()

# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # image to text
# def imgToText(url):
#     img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
#     text = img_to_text(url)[0]['generated_text']
#     return text

# # LLM
# def generate_story(scenario):
#     template = """
#                You are a story teller.
#                You can generate a short story based on a simple narrative, the story should be no more than 40 words:

#                CONTEXT: {scenario}
#                STORY:
#             """
#     prompt = PromptTemplate(template=template, input_variables=["scenario"])
#     story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo"), prompt=prompt, verbose=True)
#     story = story_llm.predict(scenario=scenario)
#     return story

# def textToSpeech(story):
#     API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
#     headers = {"Authorization": "Bearer " + HUGGINGFACEHUB_API_TOKEN}
#     payload = {"inputs": story}
#     response = requests.post(API_URL, headers=headers, json=payload)
#     with open("story.flac", "wb") as f:
#         f.write(response.content)

# def generate_story_and_play_audio(image):
#     scenario = imgToText(image.name)
#     story = generate_story(scenario)
#     textToSpeech(story)
#     return "story.flac"

# iface = gr.Interface(
#     fn=generate_story_and_play_audio,
#     inputs=gr.inputs.File(label="Upload an image"),
#     outputs=gr.outputs.Audio(label="Generated Story", type="filepath")
# )

# iface.launch()

























# # Importing Libraries
# from dotenv import find_dotenv, load_dotenv
# from transformers import pipeline
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.llms import OpenAI
# import requests
# import os

# # Loading Environment Variables
# load_dotenv(find_dotenv())
# HUGGINGFACE_HUB_API = os.getenv("HUGGINGFACE_HUB_API")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # img2text
# def img2text(url):
#     image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
#     text = image_to_text(url)[0]['generated_text']
#     print(text)
#     return text

# # generated_story
# def generated_story(sce):
#     template = """
#     You are a story teller:
#     You can generate a short story based on a simple narrative, the story should be no more than 30 words:
#     CONTEXT: {sce}
#     STORY:
#     """
#     prompt = PromptTemplate(template=template, input_variables={'sce': 'The default context'})
    
#     story_llm = LLMChain(llm=OpenAI(model_name='gpt-3.5-turbo', temperature=1), prompt=prompt, verbose=True)
    
#     story = story_llm.predict(scenario=sce)
#     print(story)
#     return story

# # text_speech
# def text_speech(message):
#     API_URL = "https://api-inference.huggingface.co/models/suno/bark"
#     headers = {"Authorization": f"Bearer {HUGGINGFACE_HUB_API}"}
#     payloads = {"input": message}
    
#     response = requests.post(API_URL, headers=headers, json=payloads)
    
#     with open('audio.flac', 'wb') as file:
#         file.write(response.content)

# # Main Code
# sce = img2text('one-direction.jpg')
# story = generated_story(sce)
# text_speech(story)
