# 📅 AI-Powered Meeting Management Tool

An end-to-end intelligent meeting assistant that automates agenda creation, live discussion tracking, and post-meeting analysis using LLMs (GPT-4), RAG (Retrieval-Augmented Generation), and speech-to-text technologies.

---

## 🚀 Features

* **Pre-Meeting Setup:**

  * Upload relevant documents and discussion points (PDF, DOCX)
  * Extract and organize key points into grouped topics
  * Automatically generate a structured meeting agenda using GPT-4

* **During Meeting:**

  * Upload meeting recording (MP3/MP4)
  * Transcribe audio using Vosk (offline speech-to-text)

* **Post-Meeting Analysis:**

  * Generate human-like, structured *Minutes of the Meeting (MoM)*
  * Use Pinecone similarity search to identify unresolved discussion points
  * Download transcript in TXT and PDF formats

---

## 🧠 Technologies Used

* **Python, Streamlit** – Frontend and orchestration
* **OpenAI GPT-4** – Summarization, agenda creation, point resolution
* **LangChain + Pinecone** – Vector embedding & semantic retrieval
* **Vosk** – Offline speech-to-text transcription
* **PyMuPDF** – PDF ingestion

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/jayashree-khairnar/meeting-management-tool.git
cd meeting-management-tool
```

2. **Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. **Download Vosk model**

* Download [vosk-model-small-en-us-0.15](https://alphacephei.com/vosk/models)
* Extract and place it in the project root directory

4. **Set environment variables** in a `.env` file:

```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

5. **Run the app**

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py             # Main Streamlit workflow
├── agenda.py          # Agenda creation & unresolved point detection
├── summary.py         # Minutes of Meeting (MoM) generation logic
├── utils.py           # Audio, PDF, file handling utilities
├── remove_files.py    # Optional: cleanup tool
├── style.css          # Custom styling
├── requirements.txt   # Dependencies
```

