# RAG Question-Answering System

A Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents using OpenAI's GPT-4 and Pinecone vector database.

The document contains a simple story about 3 children competing in a bowling competition.

## Features

- PDF text extraction and processing
- Text chunking with configurable size and overlap
- Vector embeddings using OpenAI's ada-002 model
- Semantic search using Pinecone vector database
- Question answering using GPT-4
- Detailed logging system

## Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LancemDev/gdgoc-vector-db.git
cd gdgoc-vector-db
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate 
# On fish: source/bin/activate.fish
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Project Structure

- `rag_qa.py`: Main script for processing PDFs and setting up the RAG system
- `rag_test.py`: Script for testing the question-answering system
- `assets/docs/`: Directory for storing PDF documents

## Usage

1. Place your PDF document in the `assets/docs/` directory and update the `PDF_FILE` path in `rag_qa.py`.

2. Run the indexing script:
```bash
python rag_qa.py
```
This will:
- Extract text from the PDF
- Split it into chunks
- Create embeddings
- Store them in Pinecone

3. Ask questions using the test script:
```bash
python rag_test.py
```

## Configuration

Key configuration options in the scripts:

- `PINECONE_ENV`: Your Pinecone environment (default: "us-east-1")
- `INDEX_NAME`: Name of your Pinecone index (default: "rag-demo")
- `chunk_size`: Size of text chunks (default: 500 characters)
- `overlap`: Overlap between chunks (default: 50 characters)
- `top_k`: Number of relevant chunks to retrieve (default: 3)

## How It Works

1. **Document Processing**:
   - Extracts text from PDF
   - Splits text into overlapping chunks for better context preservation

2. **Vector Storage**:
   - Generates embeddings for each text chunk using OpenAI's ada-002
   - Stores vectors in Pinecone for efficient similarity search

3. **Question Answering**:
   - Converts user question to embedding
   - Retrieves most relevant chunks from Pinecone
   - Uses GPT-4 to generate accurate answers based on retrieved context

## Logging

The system includes comprehensive logging that shows:
- PDF processing progress
- Chunk creation and storage
- Query processing
- Answer generation

Logs are formatted with timestamps and log levels for easy debugging.

## Error Handling

The system includes try-catch blocks for:
- PDF reading operations
- API calls to OpenAI
- Vector database operations
- Answer generation

## Contributing

Feel free to submit issues and pull requests.
