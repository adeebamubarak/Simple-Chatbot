import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

# Load the pre-trained Question-Answering model
qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

# Load the university prospectus text
def load_prospectus(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()
        print("Prospectus Text:", text[:500])  # Print first 500 characters for debugging
        return text



# Initialize prospectus text (ensure your file is in the same directory or update the path)
prospectus_text = load_prospectus("university_prospectus.txt")

# Function to get the answer from the model


qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def get_answer(question, context):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return "I couldn't process your question. Please try again."


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
