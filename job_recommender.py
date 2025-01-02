import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset for job recommendations
data = {
    "job_title": [
        "Data Scientist",
        "Web Developer",
        "Project Manager",
        "Software Engineer",
        "Data Analyst",
        "DevOps Engineer",
        "UX/UI Designer",
        "Machine Learning Engineer"
    ],
    "job_description": [
        "Analyze data to derive insights and build models.",
        "Develop and maintain web applications using various technologies.",
        "Oversee projects from initiation to completion, ensuring deadlines are met.",
        "Design, develop, and maintain software applications.",
        "Collect, process, and analyze data to support decision-making.",
        "Automate and streamline operations and processes in the development lifecycle.",
        "Design user-friendly interfaces and improve user experiences.",
        "Develop algorithms and models for machine learning applications."
    ],
    "skills": [
        "Python, R, SQL, Machine Learning",
        "HTML, CSS, JavaScript, React",
        "Leadership, Communication, Agile",
        "Java, C++, Software Development",
        "Excel, SQL, Data Visualization",
        "Docker, Kubernetes, CI/CD",
        "Adobe XD, Sketch, User Research",
        "Python, TensorFlow, Data Science"
    ]
}

# Create DataFrame from the dataset
job_data = pd.DataFrame(data)

# Streamlit interface
st.title("Job Recommender System")

st.write(
    "Enter your skills below, and the system will recommend jobs matching your skill set."
)

# Preprocess and vectorize job descriptions
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(job_data['job_description'])

# Define the number of clusters
k = 4  # You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=k, random_state=42)
job_data['cluster'] = kmeans.fit_predict(X)

# User input: skills
user_skills = st.text_input("Enter your skills (comma-separated):")

if user_skills:
    # Transform user input into a TF-IDF vector
    user_vector = vectorizer.transform([user_skills])

    # Predict the cluster for the user's skills
    user_cluster = kmeans.predict(user_vector)[0]

    # Get job recommendations from the predicted cluster
    recommendations = job_data[job_data['cluster'] == user_cluster]

    # Calculate similarity between the user skills and job descriptions
    recommendations['similarity'] = cosine_similarity(
        user_vector, X[recommendations.index]
    ).flatten()

    # Filter recommendations by matching skills
    user_skills_list = set(skill.strip().lower() for skill in user_skills.split(","))
    recommendations['match'] = recommendations['skills'].apply(
        lambda x: any(skill.strip().lower() in user_skills_list for skill in x.split(","))
    )

    # Keep only matching recommendations
    recommendations = recommendations[recommendations['match']]

    # Sort recommendations by similarity score
    recommendations = recommendations.sort_values(by='similarity', ascending=False)

    # Display recommendations
    if not recommendations.empty:
        st.write("Recommended Jobs for your skills:")
        st.dataframe(recommendations[['job_title', 'skills', 'job_description', 'similarity']].head(5))
    else:
        # If no exact matches, find jobs with at least one matching skill
        st.write("No exact matches found. Showing jobs with at least one matching skill:")
        all_matching_jobs = job_data[job_data['skills'].apply(
            lambda x: any(skill.strip().lower() in user_skills_list for skill in x.split(","))
        )]
        if not all_matching_jobs.empty:
            st.dataframe(all_matching_jobs[['job_title', 'skills', 'job_description']])
        else:
            st.write("No jobs found that match your skills.")
else:
    st.write("Please enter your skills to get job recommendations.")
