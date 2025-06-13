import os
import re
import logging
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from exceptions import MeetingFileError

# Load environment variables
load_dotenv()

SYSTEM_PROMPT_ORGANIZER = (
    "You are an AI assistant that organizes discussion points into clear and logical categories for any domain. "
    "Group the points by related themes, use general titles, and avoid domain-specific terms unless they naturally fit."
)

SYSTEM_PROMPT_AGENDA = (
    "You are an AI assistant that creates clear, professional, and domain-agnostic meeting agendas. "
    "The agenda must follow a logical sequence: start with general updates or high-level topics, move to technical or detailed topics, and end with wrap up and engagement points."
    "Provide time allocations, ensure clarity, and avoid any instructions or meta messages."
    "Do not include meta messages, comments, or instructions - only generate the agenda content in a clean format."
)

SYSTEM_PROMPT_COMPARISON = (
    "You are an AI assistant that checks if discussion points were addressed in a meeting transcript. "
    "Mark as 'Resolved' if discussed or actioned, even partially. Mark 'Unresolved' only if completely ignored."
)

# OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get Pinecone index
def get_pinecone_index(index_name):
    return LangchainPinecone.from_existing_index(index_name, embeddings)

# Function to read and process discussion points from a PDF
@lru_cache(maxsize=None)
def extract_and_organize_points(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        full_text = " ".join([doc.page_content for doc in documents])

        # Extract discussion points from the PDF text
        points = re.findall(r'[\n\r]+\s*(?:\d+\.|\•|\-|–|\*)\s*(.+?)(?=[\n\r]+\s*(?:\d+\.|\•|\-|–|\*)|\Z)', full_text, re.DOTALL)
        extracted_points = [point.strip() for point in points if point.strip() and len(point.split()) > 3]

        if not extracted_points:
            raise MeetingFileError("No discussion points found in the PDF.")

        # LLM prompt to organize the discussion points
        ORGANIZE_PROMPT = f"""
        Organize and group the following discussion points into logical categories based on themes or topics, regardless of domain.
        {', '.join(extracted_points)}

        For each group, provide a brief and general title (e.g., "Team Structure", "Technical Challenges", "Project Goals"). Avoid using domain-specific terminology unless it natually arises from the points.
        
        Format your response as:
        1. **Title 1**
           - Point 1
           - Point 2

        2. **Title 2**
           - Point 3
           - Point 4
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ORGANIZER},
                {"role": "user", "content": ORGANIZE_PROMPT}
            ]
        )
        organized_points = response.choices[0].message.content
        logging.info("Successfully organized discussion points")
        return organized_points, extracted_points

    except Exception as e:
        logging.error(f"Error processing discussion points: {e}")
        raise MeetingFileError(f"Failed to process discussion points: {str(e)}")

def generate_agenda(organized_points):
    try:
        AGENDA_PROMPT = f"""
        Generate a **structured and logically sequenced meeting agenda** based on the following discussion points.
        The agenda should follow a natural order of discussion:
        1. Start with introductions, high-level updates, or team structure changes.
        2. Then cover technical topics such as challenges, plans, and improvements.
        3. End with engagement, Q&A, or wrap-up topics.

        Allocate estimated time for each section proportionally based on the number and complexity of points in each group. If unsure, distribute time evenly.

        Discussion points:
        {organized_points}

        Format:
        1. **Topic 1** (XX minutes)
           - Subtopic A
           - Subtopic B

        2. **Topic 2** (XX minutes)
           - Subtopic C
           - Subtopic D

        Ensure the sequence supports a smooth discussion flow across any domain.
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_AGENDA},
                {"role": "user", "content": AGENDA_PROMPT}
            ]
        )
        generated_agenda = response.choices[0].message.content
        
        # Remove any lines that mention reviewing or adjusting the agenda
        final_agenda = '\n'.join([line for line in generated_agenda.split('\n') if "review" not in line.lower() and "adjust" not in line.lower()])
        logging.info("Successfully generated meeting agenda")
        return final_agenda
    
    except Exception as e:
        logging.error(f"Error generating agenda: {e}")
        raise MeetingFileError(f"Failed to generate agenda: {str(e)}")

def compare_points_with_transcript(discussion_points, pinecone_index):
    try:
        unresolved_points = []
        for point in discussion_points:
            # Perform similarity search for each point individually
            relevant_docs = pinecone_index.similarity_search(point, k=5)

            if not relevant_docs:
                logging.warning(f"No relevant docs found for point: {point}")
                unresolved_points.append(point)
                continue

            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            PROMPT = f"""
            Was this point addressed in the meeting?
            Point: {point}
            Transcript context:
            {context}
            
            Respond with 'Resolved' if discussed or acted upon, otherwise 'Unresolved'. Briefly explain.
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COMPARISON},
                    {"role": "user", "content": PROMPT}
                ]
            )
            
            result = response.choices[0].message.content
            if "unresolved" in result.lower():
                unresolved_points.append(point)
        return unresolved_points
    
    except Exception as e:
        logging.error(f"Error comparing points with transcript: {e}")
        raise MeetingFileError(f"Failed to compare points with transcript: {str(e)}")