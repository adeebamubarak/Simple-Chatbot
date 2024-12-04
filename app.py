import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

# Load the pre-trained Question-Answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Load the university prospectus text
def load_prospectus(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()
        print("Prospectus Text:", text[:1000])  # Print first 500 characters for debugging
        return text



# Initialize prospectus text (ensure your file is in the same directory or update the path)
prospectus_text = load_prospectus("university_prospectus.txt")

# Function to get the answer from the model

def answer_question(question, context):
    """
    This function takes a question and a context and returns a more detailed answer.
    """
    # First, get the direct answer using the QA model
    result = qa_pipeline(question=question, context=context)
    short_answer = result["answer"]
    
    # Now summarize the entire context to provide a longer answer
    summarized_text = summarize_text(context)
    
    # Combine the short answer with the summarized context for a more detailed response
    return f"Answer: {short_answer}\n\nAdditional Information: {summarized_text}"

def summarize_text(text):
    """
    This function uses a summarization model to return a detailed summary of the text.
    """
    summary = summarization_pipeline(text, max_length=500, min_length=200, do_sample=False)
    return summary[0]['summary_text']

# Function to convert text to speech and play it
def speak_text(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    return "response.mp3"

# Streamlit app interface
def main():
    st.title("University Chatbot")
    st.markdown("Ask me anything about the university, and I'll provide answers!")

    # User input for the question
    user_query = st.text_input("Enter your question:")

    if st.button("Submit"):
        # Get the answer to the user's question
        if user_query.strip() == "":
            st.warning("Please enter a valid question.")
        else:
            response = answer_question(user_query, prospectus_text)

            # Display the response
            st.subheader("Response:")
            st.write(response)

            # Generate and provide audio response
            audio_path = speak_text(response)
            audio_file = open(audio_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")

if __name__ == "__main__":
    main()
