import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define a function to transform the label values
def transform_labels(label):
    # Extract the label value
    label = label['label']
    # Map the label value to an integer value
    num = 0
    if label == -1: #'Negative'
        num = 0
    elif label == 0: #'Neutral'
        num = 1
    elif label == 1: #'Positive'
        num = 2
    # Return a dictionary with a single key-value pair
    return {'labels': num}

# Define a dictionary to map integer class labels to their meanings
class_meanings = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Example input label
label = {'label': 0}

# Call the transform_labels function and get the transformed label
transformed_label = transform_labels(label)

# Get the predicted class label and its meaning
predicted_class = transformed_label['labels']
predicted_class_meaning = class_meanings[predicted_class]

# Example confidence level
confidence_level = 1.16

# Add the "How to Use" message to the sidebar
how_to_use = """

**How to Use**
1. Enter text in the text area.
2. Click the 'Analyze' button to get the predicted sentiment of the text.
"""
st.sidebar.markdown(how_to_use)

# Define the models dictionary
models = {
    "ROBERTA": "Preencez/finetuned-Sentiment-classfication-ROBERTA-model",
}

# Define the main Streamlit app function
def main():
    st.title("Covid Tweets App")
    st.image("https://www.aimlspectrum.com/wp-content/uploads/2022/01/ser-1-624x390.png")

    menu = ["About", "Home"]  # Switched the order of "About" and "Home"
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "About":
        st.subheader("About")
        st.write("This is a sentiment analysis NLP It uses a pre-trained model to predict the sentiment of the input text.")

    else:
        st.subheader("Home")

        # Add a dropdown menu to select the model
        model_name = st.selectbox("Select a model", list(models.keys()))

        with st.form(key="nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label="Analyze")

        # Layout
        col1, col2 = st.columns(2)
        if submit_button:
            # Display balloons
            st.balloons()
            with col1:
                st.info("Results")
                tokenizer = AutoTokenizer.from_pretrained(models[model_name])
                model = AutoModelForSequenceClassification.from_pretrained(models[model_name])

                # Tokenize the input text
                inputs = tokenizer(raw_text, return_tensors="pt")

                # Make a forward pass through the model
                outputs = model(**inputs)

                # Get the predicted class and associated score
                predicted_class = outputs.logits.argmax().item()
                score = outputs.logits.softmax(dim=1)[0][predicted_class].item()

                # Compute the scores for all sentiments
                positive_score = outputs.logits.softmax(dim=1)[0][2].item()
                negative_score = outputs.logits.softmax(dim=1)[0][0].item()
                neutral_score = outputs.logits.softmax(dim=1)[0][1].item()

                # Compute the confidence level
                confidence_level = np.max(outputs.logits.detach().numpy())

                # Print the predicted class and associated score
                st.write(f"Predicted class: {predicted_class}, Score: {score:.3f}, Confidence Level: {confidence_level:.2f}")

                # Display the predicted class meaning
                st.write("Predicted Class Meaning:", class_meanings[predicted_class])

            # Create the results DataFrame
            results_df = pd.DataFrame({
                "Sentiment Class": [class_meanings[predicted_class]],
                "Score": [score]
            })

            # Create the Altair chart
            #chart = alt.Chart(results_df).mark_line().encode(
               # x="Sentiment Class",
                #y="Score",
                #color="Sentiment Class"
            #)

            # Display the chart
           #with col2:
                #st.altair_chart(chart, use_container_width=True)
                #st.write(results_df)

if __name__ == "__main__":
    main()
