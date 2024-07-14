import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import os
from gtts import gTTS
import speech_recognition as sr

# Load models
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("Franklin01/Llama-2-7b-farmingGPT-finetune")
    model = AutoModelForCausalLM.from_pretrained("Franklin01/Llama-2-7b-farmingGPT-finetune")
    translation_model_name = "facebook/nllb-200-distilled-600M"
    translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
    return tokenizer, model, translation_tokenizer, translation_model

tokenizer, model, translation_tokenizer, translation_model = load_models()

# Translation function
def translate(text, src_lang="hi", tgt_lang="en"):
    inputs = translation_tokenizer(text, return_tensors="pt", src_lang=src_lang, tgt_lang=tgt_lang)
    translated_tokens = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

# Speech-to-text function
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="hi-IN")
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Error with the request"

# Text-to-speech function
def text_to_speech(text, lang="hi"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    return "output.mp3"

# Function for processing input text
def process_text(input_text):
    translated_text = translate(input_text, src_lang="hi", tgt_lang="en")
    inputs = tokenizer(translated_text, return_tensors="pt")
    response = model.generate(**inputs)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    response_in_hindi = translate(response_text, src_lang="en", tgt_lang="hi")
    return response_text, response_in_hindi

# Function for processing uploaded audio
def process_audio(audio):
    user_speech = speech_to_text(audio)
    translated_speech = translate(user_speech, src_lang="hi", tgt_lang="en")
    inputs = tokenizer(translated_speech, return_tensors="pt")
    response = model.generate(**inputs)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    response_in_hindi = translate(response_text, src_lang="en", tgt_lang="hi")
    tts_file = text_to_speech(response_in_hindi)
    return response_text, response_in_hindi, tts_file

# Set up Gradio interface
iface = gr.Interface(
    fn=process_text,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your message in Hindi"),
    outputs=[
        gr.outputs.Textbox(label="Response in English"),
        gr.outputs.Textbox(label="Response in Hindi")
    ],
    live=True
)

audio_iface = gr.Interface(
    fn=process_audio,
    inputs=gr.inputs.Audio(source="microphone", type="filepath"),
    outputs=[
        gr.outputs.Textbox(label="Response in English"),
        gr.outputs.Textbox(label="Response in Hindi"),
        gr.outputs.Audio(label="Response in Hindi (Audio)")
    ],
    live=True
)

# Combine both interfaces
app = gr.TabbedInterface(
    [iface, audio_iface],
    ["Text Input", "Audio Input"]
)

# Launch the app
app.launch()
