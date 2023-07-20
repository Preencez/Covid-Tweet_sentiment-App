# Define the model path where the pre-trained model is saved on the Hugging Face model hub
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import sympy


# Define the model path where the pre-trained model is saved on the Hugging Face model hub
model_path = "Preencez/finetuned-Sentiment-classfication-ROBERTA-model"

# Initialize the tokenizer for the pre-trained model (using RoBERTa tokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
# Define a function to preprocess the text data
def preprocess(text):
    new_text = []
    # Replace user mentions with '@user'
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        # Replace links with 'http'
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    # Join the preprocessed text
    return " ".join(new_text)

# Define a function to perform sentiment analysis on the input text
def sentiment_analysis(text):
    # Preprocess the input text
    text = preprocess(text)

    # Tokenize the input text using the pre-trained tokenizer
    encoded_input = tokenizer(text, return_tensors='pt')

    # Feed the tokenized input to the pre-trained model and obtain output
    output = model(**encoded_input)

    # Obtain the prediction scores for the output
    scores_ = output[0][0].detach().numpy()

    # Apply softmax activation function to obtain probability distribution over the labels
    scores_ = softmax(scores_)

    # Format the output dictionary with the predicted scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l:float(s) for (l,s) in zip(labels, scores_) }

    # Return the scores
    return scores

# Define a Gradio interface to interact with the model
demo = gr.Interface(
    fn=sentiment_analysis, # Function to perform sentiment analysis
    inputs=gr.Textbox(placeholder="Write your tweet here..."), # Text input field
    outputs="label", # Output type (here, we only display the label with the highest score)
    interpretation="default", # Interpretation mode
    examples=[["Too God to be True!"]]) # Example input(s) to display on the interface

# Launch the Gradio interface
demo.launch()