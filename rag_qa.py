import os
import openai
import pinecone
import PyPDF2
import dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import logging
import time

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv.load_dotenv()

# ======= CONFIGURATION =======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"  # Change based on your Pinecone account
INDEX_NAME = "rag-demo"
PDF_FILE = "assets/docs/story.pdf"  # Your story file

# ======= SETUP =======
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete existing index if it exists
if INDEX_NAME in pc.list_indexes().names():
    logger.info(f"Deleting existing index: {INDEX_NAME}")
    pc.delete_index(INDEX_NAME)
    logger.info("Waiting for index deletion to complete...")
    time.sleep(20)  # Wait for the index to be fully deleted

# Create Pinecone index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating new Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536,  # OpenAI's ada-002 embedding dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENV
        )
    )
    logger.info("Index created successfully")
    logger.info("Waiting for index to be ready...")
    time.sleep(20)  # Wait for the index to be fully ready
else:
    logger.info(f"Using existing index: {INDEX_NAME}")

# Connect to the index
index = pc.Index(INDEX_NAME)

# ======= READ STORY FROM PDF =======
def read_pdf(file_path):
    """Extracts text from a PDF file"""
    logger.info(f"Starting to read PDF file: {file_path}")
    text = ""
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        logger.info(f"PDF has {len(reader.pages)} pages")
        for i, page in enumerate(reader.pages, 1):
            logger.info(f"Processing page {i}/{len(reader.pages)}")
            text += page.extract_text() + "\n"
    logger.info(f"Finished reading PDF. Extracted {len(text)} characters")
    return text.strip()

story_text = read_pdf(PDF_FILE)

# ======= SPLIT STORY INTO CHUNKS =======
def split_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping chunks"""
    logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={overlap})")
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

story_chunks = split_text(story_text)

# ======= EMBED AND STORE CHUNKS =======
def get_embedding(text):
    """Generates an embedding using OpenAI's Ada model"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def store_in_pinecone():
    """Embeds and uploads story chunks to Pinecone"""
    logger.info("Starting to store chunks in Pinecone")
    vectors = []
    for i, chunk in enumerate(story_chunks):
        logger.info(f"Processing chunk {i+1}/{len(story_chunks)}")
        embedding = get_embedding(chunk)
        vectors.append((f"chunk-{i}", embedding, {"text": chunk}))
    
    logger.info(f"Upserting {len(vectors)} vectors to Pinecone")
    index.upsert(vectors)
    logger.info("Successfully stored all chunks in Pinecone")

store_in_pinecone()

# ======= RETRIEVE RELEVANT CHUNKS =======
def retrieve_relevant_chunks(query, top_k=3):
    """Retrieves top K relevant text chunks from Pinecone"""
    logger.info(f"Retrieving top {top_k} chunks for query: {query}")
    query_embedding = get_embedding(query)
    
    # Update the query format
    results = index.query(
        vector=query_embedding,  # Changed from queries=[query_embedding]
        top_k=top_k,
        include_metadata=True
    )
    
    # Update the results parsing
    chunks = [match.metadata["text"] for match in results.matches]
    logger.info(f"Retrieved {len(chunks)} relevant chunks")
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Chunk {i} preview: {chunk[:100]}...")
    return chunks

# ======= GENERATE ANSWER USING GPT-4 =======
def generate_answer(query):
    """Generates answer using GPT-4 based on retrieved story chunks"""
    logger.info(f"Generating answer for query: {query}")
    relevant_texts = retrieve_relevant_chunks(query)
    context = "\n".join(relevant_texts)
    
    logger.info("Creating prompt with retrieved context")
    prompt = f"Using the provided story context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    logger.info("Sending request to GPT-4")
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that extracts answers from given text."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        logger.info("Successfully generated answer")
        logger.info(f"Answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

# ======= TEST THE SYSTEM =======
logger.info("=== Starting RAG QA System ===")
query = "Who won the competition?"
logger.info(f"Test query: {query}")
answer = generate_answer(query)
print(f"\nQuestion: {query}\nAnswer: {answer}")
logger.info("=== Finished RAG QA System ===")   