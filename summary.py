import os
from openai import OpenAI
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv()

# OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get Pinecone index
def get_pinecone_index(index_name):
    return LangchainPinecone.from_existing_index(index_name, embeddings, namespace="transcript")

# Function to retrieve relevant documents
def retrieve_documents_multi_namespace(pinecone_index, query, k=8):
    try:
        # Retrieve from transcript namespace
        transcript_docs = pinecone_index.similarity_search(query=query, k=k, namespace="transcript")

        # Retrieve from pre-meeting docs namespace
        pre_meeting_docs = pinecone_index.similarity_search(query=query, k=k//2, namespace="pre_meeting_docs")

        # Combine documents
        all_docs = transcript_docs + pre_meeting_docs
        logging.info(f"Retrieved {len(transcript_docs)} transcript docs and {len(pre_meeting_docs)} pre-meeting docs")
        return all_docs
    
    except Exception as e:
        logging.error(f"Error retrieving documents: {e}")
        
        # Fallback to single namespace if multi-namespace fails
        try:
            docs = pinecone_index.similarity_search(query=query, k=k)
            return docs
        except:
            return []
        
# Function to retrieve relevant documents
def retrieve_documents(pinecone_index, query, k=8):
    try:
        docs = pinecone_index.similarity_search(query=query, k=k)
        logging.info(f"Retrieved {len(docs)} documents")
        return docs
    except Exception as e:
        logging.error(f"Error retrieveing documents: {e}")
        return []

# Function to get chat completion
def get_chat_completion(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Prompt templates
STRUCTURED_MOM_PROMPT_TEMPLATE = """
    You are an expert meeting assistant. Create a structured Minutes of Meeting (MoM) based on the meeting transcript and pre-meeting documents provided. 

    Follow this EXACT format with each heading on a new line:

    📋 **Summary:**

    **Meeting Date**: [Extract the meeting date from the documents - look for dates in various formats like "September 14, 2022", "14/09/2022", "Sep 14, 2022", "2022-09-14", etc. If no specific date is found, note "Date not specified in documents"]

    **Participants**: [List all participants mentioned, including their roles/titles if available. Extract from both pre-meeting attendee lists and transcript mentions]

    **Agenda/Purpose**: [Summarize the main purpose based on pre-meeting objectives and actual discussion topics]

    **Topics Discussed**:
    • [First major topic with detailed explanation based on actual discussion]
    • [Second major topic with detailed explanation]
    • [Continue for all major topics that were actually covered in the meeting]

    **Decisions Made**:
    • [List specific decisions, approvals, agreements reached during the meeting]
    • [Include organizational changes, policy updates, tool adoptions, process changes]
    • [Note team restructuring, name changes, budget approvals, etc.]

    **Action Items**:
    • [Person/team responsible] to [specific action with deadline if mentioned]
    • [Continue for all action items and follow-ups identified]

    **Conclusion**: [Brief wrap-up of meeting outcomes and next steps]

    Guidelines for Date Extraction:
    - Look for dates in multiple formats: full dates, abbreviated months, numeric formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD)
    - Check both the pre-meeting document header/subject and transcript content
    - If multiple dates are mentioned, prioritize the actual meeting date over document creation dates
    - Common date patterns: "Date:", "Meeting Date:", subject lines, or dates mentioned in conversation

    Guidelines for Content Extraction:
    - Use the pre-meeting document to understand planned objectives and expected attendees
    - Extract actual discussion content from the transcript
    - Cross-reference planned vs. actual topics discussed
    - Focus on concrete outcomes: decisions made, actions assigned, agreements reached
    - Include organizational changes, policy updates, and process modifications
    - Each heading should be on its own line with proper spacing
    - Use bullet points (•) for better formatting

    Pre-meeting Document Context:
    {pre_meeting_context}

    Meeting Transcript:
    {transcript_context}

    Generate the structured MoM following the exact format above:
"""

REFINE_MOM_PROMPT_TEMPLATE = """
    You have an existing Minutes of Meeting (MoM) and additional content. Your task is to refine and expand the MoM by incorporating new information while maintaining the structured format.

    Existing MoM:
    {existing_answer}

    Additional Content:
    {context}

    Instructions:
    1. Maintain the exact format structure with each heading on a new line
    2. Add new topics, decisions, or action items discovered in the additional content
    3. Update participant lists if new names are mentioned
    4. Ensure proper spacing between sections
    5. Use bullet points (•) for lists
    6. Keep the professional, structured format

    Provide the complete refined MoM:
"""

# Functions for MoM generation
def generate_initial_mom(transcript_context, pre_meeting_context=""):
    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in creating structured Minutes of Meeting from transcripts. Always follow the exact format provided and extract accurate information from the transcript."},
        {"role": "user", "content": STRUCTURED_MOM_PROMPT_TEMPLATE.format(
            transcript_context=transcript_context,
            pre_meeting_context=pre_meeting_context
        )}
    ]
    return get_chat_completion(messages)

def refine_mom(existing_mom, additional_context):
    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in refining Minutes of Meeting. Maintain the structured format while incorporating new information accurately."},
        {"role": "user", "content": REFINE_MOM_PROMPT_TEMPLATE.format(existing_answer=existing_mom, context=additional_context)}
    ]
    return get_chat_completion(messages)

# Function to generate structured MoM using RAG
def generate_rag_summary(pinecone_index, query, pre_meeting_docs=None):
    # Retrieve documents from transcript
    retrieved_docs = retrieve_documents(pinecone_index, query, k=8)
    
    if not retrieved_docs:
        return "No relevant documents found to generate a summary."
    
    # Combine transcript content
    transcript_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Prepare pre-meeting context
    pre_meeting_context = ""
    if pre_meeting_docs:
        pre_meeting_context = "\n\n".join([doc.page_content for doc in pre_meeting_docs])
    
    # Generate structured MoM with both contexts
    mom = generate_initial_mom(transcript_context, pre_meeting_context)
    
    return mom

# Alternative function that retrieves from multiple namespaces
def generate_rag_summary_multi_namespace(pinecone_index, query):
    # Retrieve documents from both transcript and pre-meeting namespaces
    retrieved_docs = retrieve_documents_multi_namespace(pinecone_index, query, k=8)

    if not retrieved_docs:
        "No relevant documents found to generate a summary."

    # Separate transcript and pre-meeting docs
    transcript_docs = [doc for doc in retrieved_docs if doc.metadata.get('namespace')=='transcript' or 'transcript' in str(doc.metadata)]
    pre_meeting_docs = [doc for doc in retrieved_docs if doc.metadata.get('namespace')=='pre_meeting_docs' or 'pre_meeting' in str(doc.metadata)]

    # If metadata can't be separated, treat all as transcript for now
    if not transcript_docs and not pre_meeting_docs:
        transcript_docs = retrieved_docs

    # Combine content
    transcript_context = "\n\n".join([doc.page_content for doc in transcript_docs])
    pre_meeting_docs = "\n\n".join([doc.page_content for doc in pre_meeting_docs])

    # Generate structured MoM
    mom = generate_initial_mom(transcript_context, pre_meeting_docs)

    return mom

# Enhanced post-processing function for MoM
def post_process_summary(summary):
    # Remove any unwanted prefixes/suffixes
    cleaned_summary = summary.replace("Minutes of Meeting:", "").replace("MoM:", "").strip()
    
    # Ensure proper formatting
    if not cleaned_summary.startswith("📋"):
        if "📋" in cleaned_summary:
            cleaned_summary = "📋" + cleaned_summary.split("📋", 1)[1]
        else:
            cleaned_summary = "📋 **Summary:**\n" + cleaned_summary
    
    # Process lines to ensure proper spacing and formatting
    lines = cleaned_summary.split('\n')
    cleaned_lines = []

    for i, line in enumerate(lines):
        stripped_lines = line.strip()

        # Skip empty lines but preserve spacing around headers
        if not stripped_lines:
            # Only add empty line if the previous line wasn't empty
            if cleaned_lines and cleaned_lines[-1].strip():
                cleaned_lines.append('')
            continue

        # Ensure headers are on their own lines with proper spacing
        if stripped_lines.startswith('**') and stripped_lines.endswith('**'):
            # Add empty line before header if previous line isn't empty
            if cleaned_lines and cleaned_lines[-1].strip():
                cleaned_lines.append('')
            cleaned_lines.append(stripped_lines)
            cleaned_lines.append('') # Add empty line after header
        else:
            cleaned_lines.append(stripped_lines)

    # Join lines and clean up excessive empty lines
    result = '\n'.join(cleaned_lines)

    # Replace multiple consecutive lines with at most two
    import re
    result = re.sub(r'\n{3}', '\n\n', result)

    return result.strip()