# Import necessary libraries
import streamlit as st  # Streamlit for creating a web application
import io  # For handling input/output operations
from PyPDF2 import PdfReader  # To read PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split long text into smaller chunks
import os  # For interacting with the operating system
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For using Google Generative AI embeddings
import google.generativeai as genai  # Google Generative AI package
from langchain.vectorstores import FAISS  # For using FAISS vector store for embedding indexing and search
from langchain_google_genai import ChatGoogleGenerativeAI  # For using Google Generative AI model for chat
from langchain.chains.question_answering import load_qa_chain  # To load a question-answering chain
from langchain.prompts import PromptTemplate  # To create custom prompts for AI models
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load environment variables from a .env file (e.g., API keys)
load_dotenv()

# Configure the Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""  # Initialize an empty string to store extracted text
    for pdf_doc in pdf_docs:
        st.write(f"Processing file: {pdf_doc.name}")  # Display the name of the file being processed
        try:
            # Read the PDF file using PdfReader
            pdf_reader = PdfReader(io.BytesIO(pdf_doc.read()))
            # Extract text from each page of the PDF
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Append text of each page, handling cases where text might be None
        except Exception as e:
            # Display an error message if any issue occurs during PDF reading
            st.error(f"An error occurred while processing {pdf_doc.name}: {e}")
    return text  # Return the extracted text

# Function to split long text into smaller chunks for easier processing
def get_text_chunks(text):
    # Create a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # Split the text into chunks and return them
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks using embeddings
def get_vertor_store(text_chunks):
    # Use Google Generative AI embeddings for the vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create a FAISS vector store from the text chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the vector store locally for later retrieval
    vector_store.save_local('faiss_index')

# Function to create a conversational chain for question answering
def get_conversational_chain():
    # Define a prompt template for generating detailed answers
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
        if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer. 
        
        Context: \n {context}? \n
        Question: \n {question} \n

        Answer: 
    """
    # Use the Google Generative AI model for generating responses
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # Create a custom prompt with the defined template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide a response
def user_input(user_question):
    # Use embeddings for searching relevant text in the vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the existing FAISS vector store for performing a similarity search
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Search for documents similar to the user's question
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain for generating a response
    chain = get_conversational_chain()

    # Invoke the chain with the input documents and user's question, getting the response
    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    # Print the response for debugging purposes
    print(response)
    # Display the generated response to the user in the Streamlit app
    st.write("Reply: ", response["output_text"])

# Main function to set up the Streamlit app interface
def main():
    st.set_page_config("Chat with multiple PDF")  # Set the page title for the app
    st.header("Chat with multiple PDFs using Gemini")  # Display the app header

    # Get user input for asking a question related to the PDF files
    user_question = st.text_input("Ask a Question from the PDF Files")

    # If a question is provided by the user, process the input
    if user_question:
        user_input(user_question)

    # Sidebar for uploading PDF files and initiating processing
    with st.sidebar:
        st.title("Menu:")  # Sidebar title
        # Allow users to upload multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files", type="pdf", accept_multiple_files=True)
        # Button to process the uploaded PDF files
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  # Show a spinner while processing
                # Extract text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                # Split the text into smaller chunks
                text_chunks = get_text_chunks(raw_text)
                # Create a vector store from the text chunks for retrieval-based question answering
                get_vertor_store(text_chunks)
                st.success("Done")  # Indicate that processing is complete

# Entry point for the script
if __name__ == "__main__":
    main()  # Call the main function to run the app