import os
import dotenv
import logging
from pinecone import Pinecone
from openai import OpenAI

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
PINECONE_ENV = "us-east-1"
INDEX_NAME = "rag-demo"

# ======= SETUP =======
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index
index = pc.Index(INDEX_NAME)

# ======= EMBED AND RETRIEVE CHUNKS =======
def get_embedding(text):
    """Generates an embedding using OpenAI's Ada model"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieves top K relevant text chunks from Pinecone"""
    logger.info(f"Retrieving top {top_k} chunks for query: {query}")
    query_embedding = get_embedding(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
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

# ======= MAIN FUNCTION =======
if __name__ == "__main__":
    logger.info("=== Starting RAG QA Query System ===")
    query = "Who won the competition?"
    logger.info(f"Query: {query}")
    answer = generate_answer(query)
    print(f"\nQuestion: {query}\nAnswer: {answer}")
    logger.info("=== Finished RAG QA Query System ===")