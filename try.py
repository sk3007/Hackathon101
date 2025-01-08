import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from streamlit_float import *

float_init()

def text_chat_agent():
    github_token = st.secrets["GITHUB_TOKEN"]
    endpoint = "https://models.inference.ai.azure.com"
    model_name = "gpt-4o-mini"

    client = OpenAI(
        base_url=endpoint,
        api_key=github_token,
    )

    st.title("User Registration")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_name

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "Determine user MBTI and provide score of each component out of 10"
            },
            {
                "role": "system",
                "content": "Allow users to select whether they want to answer multiple choice questions or open-ended questions"
            },
            {
                "role": "system",
                "content": "Have either 8 multiple choice questions or 4 open-ended questions and ask the questions 1 by 1 while waiting for user response."
            },
            {
                "role": "system",
                "content": "Have at least 4 options for each multiple choice question"
            },
            {
                "role": "system",
                "content": "Bold out the questions and enlarge its font"
            },
            {
                "role": "system",
                "content": "Display final MBTI type with scores for each component as a list in the following order: Extroversion, Introversion, Sensing, Intuition, Feeling, Thinking, Perceiving, Judging, without explanation"
            },
            {
                "role": "assistant", 
                "content": "Hi, there! I would like to get to know you better. Do you mind if I ask you some fun questions about your preferences and how you approach different situations?"
            }
        ]

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            with open("user_response.txt", "a") as file:
                file.write(prompt + "\n")

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {
                        "role": m["role"],
                    "content": m["content"]
                    } for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)

        with open("vser_personality.txt", "w") as file:
            file.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

def voice_chat_agent():
    github_token = st.secrets["GITHUB_TOKEN"]
    endpoint = "https://models.inference.ai.azure.com"
    model_name = "gpt-4o-mini"

    client = OpenAI(
        base_url=endpoint,
        api_key=github_token,
    )

    st.title("User Registration")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_name

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "Determine user MBTI and provide score of each component out of 10"
            },
            {
                "role": "system",
                "content": "Allow users to select whether they want to answer multiple choice questions or open-ended questions"
            },
            {
                "role": "system",
                "content": "Have either 8 multiple choice questions or 4 open-ended questions and ask the questions 1 by 1 while waiting for user response."
            },
            {
                "role": "system",
                "content": "Have at least 4 options for each multiple choice question"
            },
            {
                "role": "system",
                "content": "Bold out the questions and enlarge its font"
            },
            {
                "role": "system",
                "content": "Display final MBTI type with scores for each component as a list in the following order: Extroversion, Introversion, Sensing, Intuition, Feeling, Thinking, Perceiving, Judging, without explanation"
            },
            {
                "role": "assistant", 
                "content": "Hi, there! I would like to get to know you better. Do you mind if I ask you some fun questions about your preferences and how you approach different situations?"
            }
        ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    footer_container = st.container()
    with footer_container:
        speech = audio_recorder()

    footer_container.float("bottom: 0rem;")

    if speech:
        results = pipe(speech, generate_kwargs={"language": "english"})
        transcription = results["text"]
        st.session_state.messages.append({"role": "user", "content": transcription})
        with st.chat_message("user"):
            st.markdown(transcription)
            with open("user_response.txt", "a") as file:
                file.write(transcription + "\n")

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {
                        "role": m["role"],
                    "content": m["content"]
                    } for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)

        with open("vser_personality.txt", "w") as file:
            file.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if "page" not in st.session_state:
    st.session_state.page = None  

def navigate(page_name):
    st.session_state.page = page_name
    st.rerun()

if st.session_state.page == "text_page":
    text_chat_agent()
elif st.session_state.page == "voice_page":
    voice_chat_agent()
else:
    st.title("User Registration")
    st.write("Please choose either one of the following communication methods to proceed.")
    
    if st.button("Text", use_container_width=True):
        navigate("text_page")
    if st.button("Voice", use_container_width=True):
        navigate("voice_page")