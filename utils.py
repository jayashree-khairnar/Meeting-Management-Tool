import os
import tempfile
from typing import List
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from typing import Union
from langchain_core.documents import Document
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import wave
from fpdf import FPDF
import json
import shutil

# Load environment variables
load_dotenv()

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def handle_uploaded_file(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def extract_audio_from_video(video_path):
    # Extract audio from video
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    audio.write_audiofile(audio_path)
    video.close()
    audio.close()
    return audio_path

def split_audio(audio_file_path, chunk_length_ms=60000):
    try:
        audio = AudioSegment.from_wav(audio_file_path)
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        # Create a temporary folder to store chunks
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []

        for idx, chunk in enumerate(chunks):
            chunk = chunk.set_channels(1)  # Convert the chunk to mono
            chunk_name = os.path.join(temp_dir, f"chunk{idx}.wav")
            chunk.export(chunk_name, format='wav')
            chunk_paths.append(chunk_name)

        return chunk_paths, temp_dir
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return [], None

def transcribe_audio_chunks_vosk(chunk_paths, vosk_model_path):
    model = Model(vosk_model_path)
    full_transcription = ""

    for chunk_path in chunk_paths:
        try:
            wf = wave.open(chunk_path, "rb")
            rec = KaldiRecognizer(model, wf.getframerate())
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    full_transcription += result.get("text", "") + " "
            wf.close()

        except Exception as e:
            print(f"Error during transcription of {chunk_path}: {e}")

    return full_transcription.strip()

def write_transcription_to_txt(transcription, output_txt_path):
    try:
        with open(output_txt_path, "w") as txt_file:
            txt_file.write(transcription)
        print(f"Transcription written to .txt at: {output_txt_path}")
    except Exception as e:
        print(f"Error writing transcription to .txt file: {e}")

def convert_txt_to_pdf(txt_file_path, output_pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    try:
        with open(txt_file_path, "r") as txt_file:
            for line in txt_file:
                pdf.multi_cell(0, 10, line)

        pdf.output(output_pdf_path)
        print(f"Transcription converted to PDF at: {output_pdf_path}")
    except Exception as e:
        print(f"Error converting .txt file to PDF: {e}")

def convert_audio_to_text(audio_path, vosk_model_path):
    # Extract audio if it's a video file
    if audio_path.endswith('.mp4'):
        audio_path = extract_audio_from_video(audio_path)

    # Split audio into chunks
    chunk_paths, temp_dir = split_audio(audio_path)

    # Transcribe audio chunks
    if chunk_paths:
        transcription = transcribe_audio_chunks_vosk(chunk_paths, vosk_model_path)

        # Write transcription to a .txt file
        txt_output_path = audio_path.rsplit('.', 1)[0] + '.txt'
        write_transcription_to_txt(transcription, txt_output_path)

        # Convert .txt file to a PDF file
        pdf_output_path = audio_path.rsplit('.', 1)[0] + '.pdf'
        convert_txt_to_pdf(txt_output_path, pdf_output_path)

        # Clean up temporary directory
        if temp_dir:
            shutil.rmtree(temp_dir)
            print(f"Temporary folder {temp_dir} deleted.")

        return txt_output_path, pdf_output_path
    else:
        print("No chunks to transcribe.")
        return None, None

def read_and_process_doc(file_path: str, chunk_size=1000, chunk_overlap=100) -> List[str]:
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def initialize_pinecone():
    # Initialize Pinecone
    return Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

def get_or_create_pinecone_index():
    pc = initialize_pinecone()
    index_name = "meeting-embeddings"

    # Create the index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings are 1536 dimensions
            metric='cosine',
            spec=ServerlessSpec(
                cloud=os.getenv('PINECONE_CLOUD', 'aws'),
                region=os.getenv('PINECONE_REGION', 'us-west-2')
            )
        )

    return index_name

def upsert_to_pinecone(documents: List[Union[Document, dict]], namespace: str = "default"):
    index_name = get_or_create_pinecone_index()
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Convert dictionaries to Document objects if necessary
    processed_documents = []
    for doc in documents:
        if isinstance(doc, dict):
            processed_documents.append(Document(page_content=doc['page_content'], metadata=doc.get('metadata', {})))
        else:
            processed_documents.append(doc)
    
    # Initialize LangchainPinecone with the index name
    vectorstore = LangchainPinecone.from_documents(
        processed_documents, 
        embeddings, 
        index_name=index_name,
        namespace=namespace
    )
    
    return vectorstore