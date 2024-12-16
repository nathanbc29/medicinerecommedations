import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the medicine data
file_path = 'Medicine_Details.csv'
data = pd.read_csv(file_path)

# Preprocessing: Focus on 'Uses' for symptom matching
data['Uses'] = data['Uses'].fillna('').str.lower()
data['Medicine Name'] = data['Medicine Name'].fillna('Unknown')

# TF-IDF Vectorizer for symptom matching
vectorizer = TfidfVectorizer(stop_words='english')
uses_matrix = vectorizer.fit_transform(data['Uses'])

def recommend_medicine(symptoms):
    """Recommends medicines or suggests seeing a doctor based on symptoms."""
    symptoms = symptoms.lower()
    symptom_vector = vectorizer.transform([symptoms])

    # Calculate similarity
    similarity_scores = cosine_similarity(symptom_vector, uses_matrix).flatten()

    # Get top matches
    top_indices = similarity_scores.argsort()[-5:][::-1]
    top_scores = similarity_scores[top_indices]

    # If the highest score is too low, suggest seeing a doctor
    if max(top_scores) < 0.1:
        return "Your symptoms seem severe or unclear. Please consult a doctor."

    # Filter recommendations based on accuracy threshold
    recommendations = []
    for idx, score in zip(top_indices, top_scores):
        if score > 0:
            accuracy = score * 100
            if accuracy >= 60:
                recommendations.append({
                    'Medicine Name': data.iloc[idx]['Medicine Name'],
                    'Uses': data.iloc[idx]['Uses'],
                    'Side_effects': data.iloc[idx]['Side_effects'],
                    'Manufacturer': data.iloc[idx]['Manufacturer'],
                    'Accuracy': f"{accuracy:.2f}%"
                })

    if not recommendations:
        return "The matches found have low accuracy. Please consult a doctor."

    return recommendations

# Streamlit app
st.title("Symptom-Based Medicine Recommendation System")
st.write("Input your symptoms below to get medicine recommendations or a suggestion to consult a doctor.")

# User input
symptoms_input = st.text_input("Enter your symptoms:", "")

if symptoms_input:
    result = recommend_medicine(symptoms_input)

    if isinstance(result, str):
        st.warning(result)
    else:
        st.success("Here are the recommended medicines based on your symptoms:")
        for rec in result:
            st.write(f"**Medicine Name**: {rec['Medicine Name']}")
            st.write(f"**Uses**: {rec['Uses']}")
            st.write(f"**Side Effects**: {rec['Side_effects']}")
            st.write(f"**Manufacturer**: {rec['Manufacturer']}")
            st.write(f"**Accuracy**: {rec['Accuracy']}")
            st.write("---")
