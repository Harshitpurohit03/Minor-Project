import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

def predict_sentiment(review, model, scaler, vectorizer):
  """Predicts the sentiment of a product review.

  Args:
      review (str): The product review text.
      model: The loaded sentiment analysis model.
      scaler: The loaded data scaler.
      vectorizer: The loaded TF-IDF vectorizer.

  Returns:
      str: The predicted sentiment ("Positive" or "Negative").
  """

  try:
    review_vector = vectorizer.transform([review])
    review_scaled = scaler.transform(review_vector.toarray())
    prediction = model.predict(review_scaled)[0]

    if prediction == 0:
      return "Negative Review"
    else:
      return "Positive Review"
  except Exception as e:  # Catch generic exceptions for debugging
    return f"An error occurred: {str(e)}"

# Load the sentiment analysis model, scaler, and vectorizer (error handling added)
try:
  model = pk.load(open('model.pkl','rb'))
  scaler = pk.load(open('scaler.pkl','rb'))
  vectorizer = TfidfVectorizer(decode_error='strict')  # Handle potential encoding issues
except FileNotFoundError:
  st.error("Error: Model files (model.pkl, scaler.pkl) not found!")
except Exception as e:  # Catch generic exceptions for debugging model loading
  st.error(f"An error occurred while loading the model: {str(e)}")

# Set a descriptive page title and favicon
st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon=":star:"
)

# Display a clear title and instructions for the user
st.title("Product Review Sentiment Analysis")
st.write("Enter a product review and click 'Predict' to analyze its sentiment.")

# Get user input for the product review
review = st.text_input('Enter Product Review:')

# Display a button to trigger prediction
if st.button('Predict'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.write('Negative Review')
    else:
        st.write('Positive Review')
    
