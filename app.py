import streamlit as st
import logging
from utils import handle_uploaded_file, read_and_process_doc, upsert_to_pinecone, load_css, convert_audio_to_text
from summary import generate_rag_summary, post_process_summary
from agenda import extract_and_organize_points, generate_agenda, compare_points_with_transcript, get_pinecone_index
import os

# Set up logging
logging.basicConfig(filename='app.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Set page config as the first Streamlit command
st.set_page_config(page_title="Meeting Management Tool", page_icon="📅", layout="wide")

# Load custom CSS
load_css('style.css')

# Streamlit UI for Meeting Management Tool
st.title("📅 Meeting Management Tool")
st.write("""
    Welcome to the Meeting Management Tool! This powerful application helps you streamline your meetings by:
    - Organizing pre-meeting documents
    - Generating structured meeting agendas
    - Tracking meeting progress
    - Creating concise meeting summaries
    - Identifying unresolved discussion points
""")

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 'pre_meeting'

# Pre-meeting stage
if st.session_state.stage == 'pre_meeting':
    st.header("📋 Pre-Meeting Setup")
    
    # Step 1: Upload Pre-Meeting Documents
    st.subheader("1. Upload Pre-Meeting Documents")
    uploaded_docs = st.file_uploader("Upload relevant documents (PDF, DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    
    # Step 2: Upload Discussion Points PDF
    st.subheader("2. Upload Discussion Points PDF")
    discussion_points_file = st.file_uploader("Upload Discussion Points (PDF)", type=["pdf"])
    
    if st.button("Generate Agenda"):
        if discussion_points_file and uploaded_docs:
            with st.spinner("Generating agenda and processing documents..."):
                # Save uploaded files
                doc_paths = [handle_uploaded_file(doc) for doc in uploaded_docs]
                discussion_points_path = handle_uploaded_file(discussion_points_file)
          
                # Process and embed pre-meeting documents
                pre_meeting_docs = []
                for doc_path in doc_paths:
                    documents = read_and_process_doc(doc_path)
                    pre_meeting_docs.extend(documents)
                    upsert_to_pinecone(documents, namespace="pre_meeting_docs")
                
                st.session_state.pre_meeting_docs = pre_meeting_docs

                # Extract and organize discussion points
                organized_points, extracted_points = extract_and_organize_points(discussion_points_path)
                
                # Embed discussion points
                discussion_point_docs = [{"page_content": point, "metadata": {"type": "discussion_point"}} for point in extracted_points]
                upsert_to_pinecone(discussion_point_docs, namespace="discussion_points")
                
                if organized_points:
                    # Generate Agenda
                    agenda = generate_agenda(organized_points)
                    st.session_state.agenda = agenda
                    st.session_state.extracted_points = extracted_points
                    st.session_state.stage = 'during_meeting'
                    st.rerun()
                else:
                    st.error("Failed to organize discussion points.")
        else:
            st.error("Please upload both the Discussion Points PDF and pre-meeting documents to generate an agenda.")

# During-meeting stage
elif st.session_state.stage == 'during_meeting':
    st.header("🎯 Meeting In Progress")
    
    # Display Agenda
    st.subheader("Meeting Agenda:")
    st.write(st.session_state.agenda)
    
    # Option to upload meeting recording
    st.subheader("Upload Meeting Recording")
    uploaded_audio = st.file_uploader("Upload the Meeting Recording (MP3 or MP4)", type=["mp3", "mp4"])
    
    if st.button("End Meeting and Process"):
        if uploaded_audio:
            st.session_state.uploaded_audio = uploaded_audio
            st.session_state.stage = 'post_meeting'
            st.rerun()
        else:
            st.error("Please upload the meeting recording before ending the meeting.")

# Post-meeting stage
elif st.session_state.stage == 'post_meeting':
    st.header("📊 Post-Meeting Analysis")
    
    with st.spinner("Processing meeting data..."):
        # Handle uploaded file
        recording_path = handle_uploaded_file(st.session_state.uploaded_audio)
        
        # Convert audio to text
        vosk_model_path = "vosk-model-small-en-us-0.15"  # Ensure this path is correct
        txt_path, pdf_path = convert_audio_to_text(recording_path, vosk_model_path)
        
        if txt_path:
            # Process the transcript
            transcript_docs = read_and_process_doc(txt_path)
            transcript_text = "\n".join([doc.page_content for doc in transcript_docs])
            
            # Get the pre-meeting docs from session state
            pre_meeting_docs = st.session_state.get('pre_meeting_docs', [])
            pre_meeting_text = "\n".join([doc.page_content for doc in pre_meeting_docs])
            full_context = f"{pre_meeting_text}\n\n{transcript_text}"

            # Upsert to Pinecone
            pinecone_index = upsert_to_pinecone(transcript_docs, namespace="transcript")
            
            # Debug: Check Pinecone has transcript
            sample_docs = pinecone_index.similarity_search("test", k=5)
            for i, doc in enumerate(sample_docs):
                print(f"Sample {i+1}: {doc.page_content[:200]}")
            
            SUMMARY_QUERY = "meeting discussion agenda topics decisions action items participants"

            # Generate Summary using the RAG 
            summary = generate_rag_summary(pinecone_index, SUMMARY_QUERY, pre_meeting_docs)
            if summary:
                final_summary = post_process_summary(summary)
                st.subheader("Meeting Summary:")
                st.write(final_summary)
            
            # Identify Unresolved Points
            unresolved_points = compare_points_with_transcript(st.session_state.extracted_points, pinecone_index)
            st.subheader("Unresolved Points:")
            if unresolved_points:
                for point in unresolved_points:
                    st.write(f"- {point}")
            else:
                st.write("All points were resolved in the meeting.")
            
            # Provide download links for transcript files
            st.subheader("Meeting Transcript:")
            st.download_button(
                label="Download Transcript (TXT)",
                data=open(txt_path, 'rb'),
                file_name="meeting_transcript.txt",
                mime="text/plain"
            )
            st.download_button(
                label="Download Transcript (PDF)",
                data=open(pdf_path, 'rb'),
                file_name="meeting_transcript.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Failed to transcribe the audio file.")
    
    st.success("Meeting processing completed!")
    if st.button("Start New Meeting"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Add a footer
st.markdown("---")
st.markdown("© 2025 Meeting Management Tool. All rights reserved.")