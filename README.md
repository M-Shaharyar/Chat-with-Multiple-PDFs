# Chat with PDF

A Streamlit application that allows users to chat with the content of multiple PDF files using Google Generative AI. Users can upload PDF documents, and the application will process them to answer questions based on the extracted text.

## Features

- Upload multiple PDF files.
- Extract text from the PDFs.
- Split the extracted text into manageable chunks.
- Utilize embeddings to enable efficient searching of relevant text.
- Ask questions and receive detailed answers based on the PDF content.

## Requirements

Ensure you have the following packages installed:

- Streamlit
- PyPDF2
- langchain
- langchain-google-genai
- python-dotenv

You can install the required packages using:

```
pip install -r requirements.txt
```
## Environment Variables
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Clone the Repository Locally
```
git clone https://github.com/M-Shaharyar/Chat-With-PDF.git
```
## Add Files
```
cd Chat-With-PDF
```
## Running the Application
```
streamlit run app.py
```

## Usage

1. **Upload PDF Files**: Use the file uploader in the sidebar to select and upload one or more PDF files.
2. **Process PDF Files**: After uploading, click the **Submit & Process** button. The application will extract text from the PDFs and create a vector store for searching.
3. **Ask Questions**: Enter your questions related to the content of the uploaded PDFs in the input box. The application will respond with answers based on the extracted information.


