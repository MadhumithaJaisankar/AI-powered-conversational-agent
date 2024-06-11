import os
import io
import streamlit as st
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pptx import Presentation
import requests
from bs4 import BeautifulSoup
import shutil
from googletrans import Translator
from pytube import YouTube
import pandas as pd
from pytube import Playlist
from langdetect import detect
import json
import numpy as np
import main
from flask import Flask, request, jsonify
from datetime import datetime
import requests
import json
from oauth2client.service_account import ServiceAccountCredentials
import logging
import pandas as pd
import subprocess

load_dotenv()
app = Flask(__name__)

# Set the API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

VERSION="v19.0"
PHONE_NUMBER_ID="271322246064973"
recipients = ['+919597315025','+917010922948']
verify_token ="12345"
access_token = "EAALJ6vKZAwAQBO9vHj1ZAFjk1ZAQm1IfqTKze9qRbnCiWhBGQuLW8UCXIOabDvVAZCbzDCAkNhzQFMO8W8bnZBqazIEVYYTU7KmJaynz6e6OvVRxlzw55tPGO98GrS8lUL5d4oDzoZCbhlD8aE1lyLpAnSVfdUIwfJsIyTfrIvIc3IlVFJW3ZC4z6hhtdIcI3fnGfb6aciK2FgXMiZAHlRQZD"

import pickle

# Function to save vector store to file
def save_vector_store(vector_store, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vector_store, f)

# Function to load vector store from file
def load_vector_store(filename):
    with open(filename, 'rb') as f:
        vector_store = pickle.load(f)
    return vector_store

# Function to preprocess CSV files and cluster data using RAG engine
def preprocess_and_cluster(df):
    if 'Text' in df.columns:
        text_chunks = get_text_chunks(' '.join(df['Text'].astype(str)))
        return text_chunks
    else:
        return []

@app.route('/webhook', methods=['GET'])
def webhook_get():
    return verify()

@app.route('/webhook', methods=['POST'])
def webhook_post():
    return handle_message()

def verify():
    # Parse params from the webhook verification request
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    expected_token = verify_token
    logging.info(f"Received token: {token}, expected token: {expected_token}")
    # Check if a token and mode were sent
    if mode and token:
        # Check the mode and token sent are correct
        if mode == "subscribe" and token == expected_token:
            # Respond with 200 OK and challenge token from the request
            logging.info("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            # Responds with '403 Forbidden' if verify tokens do not match
            logging.info("VERIFICATION_FAILED")
            return jsonify({"status": "error", "message": "Verification failed"}), 403
    else:
        # Responds with '400 Bad Request' if verify tokens do not match
        logging.info("MISSING_PARAMETER")
        return jsonify({"status": "error", "message": "Missing parameters"}), 400
    

# Function to split text into chunks for vectorization
def get_text_chunks(text):
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def check_faiss_index_data(vector_store):
    try:
        # Get FaissIndex
        faiss_index = vector_store.index
        # Get number of vectors
        num_vectors = faiss_index.ntotal
        # Get dimensionality of vectors
        dim = faiss_index.d
        # Get a random vector from the index
        random_vector_id = np.random.randint(0, num_vectors)
        random_vector = faiss_index.reconstruct(random_vector_id)
        
        # Display information
        st.write("Number of vectors in FaissIndex:", num_vectors)
        st.write("Dimensionality of vectors:", dim)
        st.write("Example random vector from FaissIndex:", random_vector)
    except Exception as e:
        st.error(f"Error checking FaissIndex data: {str(e)}")



### Adding Whatsapp Functionality

def send_whatsapp_message(recipients, message):
    for to in recipients:
        url = f"https://graph.facebook.com/{VERSION}/{PHONE_NUMBER_ID}/messages"
        headers = {'Content-Type': 'application/json','Authorization': f'Bearer {access_token}'}
        data = {
            'recipient_type': 'individual',
            'to': to,
            'type': 'text',
            'text': {
                'body': message
            },
            'messaging_product': 'whatsapp'
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print('Message sent:', response.json())
    return response



def handle_message():
    body = request.get_json()
    # logging.info(f"request body: {body}")
    print(body)
    # Check if it's a WhatsApp status update
    if (
        body.get("entry", [{}])[0]
        .get("changes", [{}])[0]
        .get("value", {})
        .get("statuses")
    ):
        logging.info("Received a WhatsApp status update.")
        return jsonify({"status": "ok"}), 200

    try:
        #if is_valid_whatsapp_message(body):
            process_message(body)
            return jsonify({"status": "ok"}), 200
        #else:
            # if the request is not a WhatsApp API event, return an error
            #return (
                #jsonify({"status": "error", "message": "Not a WhatsApp API event"}),
                #404,
            #)
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON")
        return jsonify({"status": "error", "message": "Invalid JSON provided"}), 400

def process_message(body):
    message = request.json
    from_ =  body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    message_body = message["text"]["body"]
    #generate a response
    reply = generate_reply(message_body)
    #send a response back
    send_whatsapp_message([from_], reply)
    return '', 200


def generate_reply(message):
    vector_store = load_vector_store("all_data_vector_store.pkl")
    try:
        prompt_template = """
        Summarize the context when asked. Answer the question as if you were Modi Ji, drawing primarily from the provided context. Ensure to include a concise summary and all relevant details related to the question. Behave as much like Modi Ji as possible and rely heavily on the provided context for your response. Avoid giving answers that are unrelated to the context.".\n\n
        Context:\n {context}?\n
        Question:\n {question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", 'question'])
        conversational_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Error loading conversational chain: {str(e)}")
        conversational_chain = None
    
    answer = handle_user_input(message, vector_store, conversational_chain)
    return answer
    

# Inside the main function
#check_faiss_index_data(vector_store)

def main():
    try:
        st.set_page_config(page_title="Chatbot Emulator", layout='wide', initial_sidebar_state='auto')
        st.title("Chatbot Emulator")
        
        # Load vector store from file
        vector_store = load_vector_store("all_data_vector_store.pkl")
        
        if vector_store:
            try:
                st.success("Data processed successfully!")
                
            except Exception as e:
                st.error(f"Error loading vector stores: {str(e)}")
                return

            # Load conversational chain
            try:
                prompt_template = """
                Summarize the context when asked. Answer the question as if you were Modi Ji, drawing primarily from the provided context. Ensure to include a concise summary and all relevant details related to the question. Behave as much like Modi Ji as possible and rely heavily on the provided context for your response. Avoid giving answers that are unrelated to the context.".\n\n
                Context:\n {context}?\n
                Question:\n {question}\n

                Answer:
                """
                model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3, google_api_key=GOOGLE_API_KEY)
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", 'question'])
                conversational_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            except Exception as e:
                st.error(f"Error loading conversational chain: {str(e)}")
                conversational_chain = None

            # Chat functionality
            user_question = st.text_input("Ask a question:")
            if st.button("Ask"):
                if user_question:
                    if conversational_chain:
                        # Process user input
                        answer = handle_user_input(user_question, vector_store, conversational_chain)
                        st.write(f"Reply:", answer)
                    else:
                        st.error("Error processing user input.")
            
            label = "Enable WhatsApp"
            is_enabled = st.checkbox(label)
            
            if is_enabled:
                pass
                #st.write("WhatsApp is enabled. Switching to Flask application...")
                #send_whatsapp_message(recipients, message="Welcome to Modi Bot 1.0. I am trained on several official videos, Tweets, Information on the website and other first party data source of Prime Minister Modi. You can ask me about questions about myself , my policies, my key decision etc.")
                #if user_question:
                    #send_whatsapp_message(recipients, message="You asked "+user_question)
                    #send_whatsapp_message(recipients, message=answer)
                    ## Webhook to receive and process a WhatsApp message
                    #@app.route('/webhook', methods=['POST'])
                    #def webhook_post():
                     #   return handle_message()
                
                    #app.run(host="0.0.0.0",port=5000)
                    #handle_message()
            else:
                send_whatsapp_message(recipients, message="Welcome to Modi Bot 1.0. I am trained on several official videos, Tweets, Information on the website and other first party data source of Prime Minister Modi. You can ask me about questions about myself , my policies, my key decision etc.")
                pass                  
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Function to handle user input
def handle_user_input(user_question, vector_store, conversational_chain):
    try:
        # Perform similarity search using the vector store
        docs = vector_store.similarity_search(user_question)
        # Generate response using the conversational chain
        response = conversational_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    except Exception as e:  
        st.error(f"Error processing user input: {str(e)}")
    return response["output_text"]

#send_whatsapp_message(recipients, message="Welcome to Modi Bot 1.0. I am trained on several official videos, Tweets, Information on the website and other first party data source of Prime Minister Modi. You can ask me about questions about myself , my policies, my key decision etc.")
if __name__ == "__main__":
    main()
    app.run(host='0.0.0.0', port=5000)