import os
import dotenv
import logging
from openai import OpenAI  # Updated import

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

# ======= SETUP ======= 
client = OpenAI(api_key=OPENAI_API_KEY)  # Updated client initialization

# ======= SIMPLE CHATBOT =======
def chat_with_gpt(prompt):
    """Generates a response using GPT-4"""
    logger.info(f"Generating response for prompt: {prompt}")
    try:
        response = client.chat.completions.create(  # Updated API call
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        logger.info("Successfully generated response")
        logger.info(f"Response: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

# ======= MAIN FUNCTION =======
if __name__ == "__main__":
    logger.info("=== Starting Simple Chatbot ===")
    print("Welcome to the chatbot! Type 'exit' or 'quit' to end the conversation.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting chatbot")
                break
            response = chat_with_gpt(user_input)
            print(f"\nBot: {response}")
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"\nAn error occurred: {str(e)}")
            continue
    logger.info("=== Finished Simple Chatbot ===")