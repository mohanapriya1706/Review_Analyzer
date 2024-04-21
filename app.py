# sudo apt-get install build-essential
import streamlit as st
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
import tensorflow_text
from transformers import pipeline
import nltk
import emoji
import google.generativeai as genai

from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
# nltk.download('punkt')
# nltk.download('stopwords')


generation_config = {
  "candidate_count": 1,
  "max_output_tokens": 256,
  "temperature": 1.0,
  "top_p": 0.7,
}

safety_settings=[
  {
    "category": "HARM_CATEGORY_DANGEROUS",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]

genai.configure(api_key="Enter Your API key Here")
llm = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
bertmodel = BERTopic(verbose=True,vectorizer_model=vectorizer_model,embedding_model='paraphrase-MiniLM-L3-v2', min_topic_size= 7)
#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

@st.cache_resource()
def aimodel():
    import tensorflow_text
    import tensorflow_hub as hub
    loaded_model = load_model("Aimodel.h5", custom_objects={'KerasLayer': hub.KerasLayer})
    return loaded_model

@st.cache_resource()
def ml(x):
    st.session_state.loaded_model = aimodel()
    prediction = st.session_state.loaded_model.predict([x])
    prediction = (prediction  > 0.5).astype(int)
    if prediction ==[0]:
        return "Human"
    else:
        return "AI"

@st.cache_resource()
def sentiment_analyis(text, return_max_label=True):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(encoded_text['input_ids'], encoded_text['attention_mask'])
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    max_index = scores.argmax()
    max_label = labels[max_index]
    max_score = scores[max_index]

    if return_max_label:
        return max_label
    else:
        return round(max_score, 3)


@st.cache_resource()
def analysis():
    data = st.session_state.df.copy()
    data["Sentiment"] = data["Reviews"].apply(lambda x: sentiment_analyis(x,return_max_label=True))
    data["Sentiment %"] = data["Reviews"].apply(lambda x: sentiment_analyis(x, return_max_label=False))
    st.session_state.data = data
    return st.session_state.data

def clear_cache_and_session():
    st.cache_resource.clear()
    st.session_state.clear()

st.set_page_config(page_title="Reviews analysis", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stDecoration"]{
        display: none;
    }
    </style>
"""
,unsafe_allow_html=True
)
st.title("Review Sentiment Analysis")

option = option_menu(None, ["Sentiment", "Visual Analysis", 'Words Distribution',"Reviews Authenticity","Topic Modelling"],
                     icons=['emoji-smile-upside-down', "images", 'alphabet',"fingerprint","diagram-3"],
                     menu_icon="cast", default_index=0, orientation="horizontal",)

def clicked(button):
    st.session_state.clicked = {}  # Initialize the clicked attribute
    st.session_state.clicked[button] = True

with st.sidebar:
    csv = st.file_uploader("Upload a CSV file containing reviews:😃☹️😑", type=["csv"])
    if st.button("Submit & Process", on_click=clicked, args=[1]):
        with st.spinner("Processing..."):
            if csv is not None:
                df = pd.read_csv(csv)
                st.session_state.df = df
                st.success("File Uploaded", icon="🎉")
    st.button("Refresh",on_click=clear_cache_and_session)


def sentiment():
    st.table(st.session_state.df.head(5))
    if st.button("Click to get Sentiments of your Data",on_click=clicked,args=[1]):
        if 'data' not in st.session_state:
            with st.spinner("Analyzing the sentiments of your Data Processing..."):
                st.session_state.data = analysis()
            
        with st.spinner("Almost done Please wait"):  
            st.header("Sentiments of your data")
            st.dataframe(st.session_state.data.head(5),width=1500)
        st.success("Sentiments for your Dataset were Analyzed", icon="🎉")



def pie_chart(pie_chart_analysis):
    sentiment_counts = pie_chart_analysis["Sentiment"].value_counts()
    total_reviews = len(pie_chart_analysis)
    positive_percentage = (sentiment_counts.get('Positive', 0) / total_reviews) * 100
    negative_percentage = (sentiment_counts.get('Negative', 0) / total_reviews) * 100
    neutral_percentage = (sentiment_counts.get('Neutral', 0) / total_reviews) * 100

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive_percentage, negative_percentage, neutral_percentage]

    st.header("Overall Aggregriate Reviews Percentage")
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes)])
    fig.update_layout(title="Sentiment Analysis", width=800, height=600)
    st.plotly_chart(fig,use_container_width=True)

    st.header("Occurances of Reviews")
    colors = ['#05a823', '#a80520', '#6a54e8']
    value_counts = pie_chart_analysis["Sentiment"].value_counts()
    fig2 = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values,marker_color=colors)])
    fig2.update_layout(title="Sentiment Analysis", xaxis_title="Sentiment", yaxis_title="Counts")
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)



def cloud():
    #plot word_cloud
    all_reviews = ' '.join(st.session_state.data['Reviews']).lower()

    tokens = word_tokenize(all_reviews)
    filtered_words = [word for word in tokens if word not in stopwords.words('english') and word.isalpha()]

    filtered_text = ' '.join(filtered_words)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(filtered_text)
    fig, ax = plt.subplots(figsize=(12,6), facecolor = None)
    ax.imshow(wordcloud)
    ax.axis('off')
    st.pyplot(fig)


    # Count and display most common words
    word_counts = Counter(filtered_words)

    st.header("Most Common Words in Reviews")
    most_common_words = word_counts.most_common(st.number_input("Give input for Most common Word Counts", step=1,min_value=1,value=10))
    #st.write(most_common_words)

    #pie chart analysis
    selected_word = st.selectbox("Select a word for sentiment analysis:", [word.title() for word,_ in most_common_words])
    
    st.session_state.filtered_reviews = st.session_state.data[st.session_state.data['Reviews'].str.lower().str.contains(selected_word.lower())]
    st.write(st.session_state.filtered_reviews.shape)
    st.dataframe(st.session_state.filtered_reviews.sample(5),width=1500)
    filter = st.session_state.filtered_reviews.copy()
    pie_chart(filter)




@st.cache_resource()
def aunthenticity():
    st.header("Analyzing the aunthenticty of the reviews by Analyzing it is AI Genereated or Not")
    st.dataframe(st.session_state.data[["Reviews","Sentiment","Sentiment %"]].sample(5),width=1500)
    auth = st.session_state.data.copy()
    #auth.drop(columns=['Timeframe'], inplace=True)
    auth["Data_context"] = auth["Reviews"].apply(lambda x : ml(x))
    st.session_state.auth = auth
    st.header("Nature of the data")
    st.dataframe(st.session_state.auth.head(5),width=1500)

    st.header("Pie Chart for The reviews Authenticity")
    counts = st.session_state.auth["Data_context"].value_counts()

    colors = ['green', 'red']
    trace = go.Pie(labels=counts.index, values=counts.values, hole=0.3,marker=dict(colors=colors))
    layout = go.Layout(title='Data Context Distribution',width=800, height=600)
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig,use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AI generated Reviews")
        st.dataframe(st.session_state.auth[st.session_state.auth['Data_context'] == "AI"])
    with col2:
        st.subheader("Human Generated Reviews")
        st.dataframe(st.session_state.auth[st.session_state.auth['Data_context'] == "Human"])
    return st.session_state.auth

@st.cache_resource()
def topic(top_n_topics):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.dataframe(st.session_state.auth.head(5),width=1500)
    topic_df = st.session_state.auth.copy()
    with st.spinner("Genearting Topics....."):
        topic_df['Reviews'] = topic_df['Reviews'].apply(lambda x: emoji.demojize(x, delimiters=("", "")))
        headline_topics, probs = bertmodel.fit_transform(topic_df.Reviews)
    freq = bertmodel.get_topic_info()
    num_topics = "Number of topics: {}".format(len(freq))
    st.write(num_topics)
    st.dataframe(freq.head(),width=1500)

    st.header("Topic Word Scores")
    fig1 = bertmodel.visualize_barchart(top_n_topics=top_n_topics)
    st.plotly_chart(fig1,use_container_width=True)
    st.header("Interoptic Distance Map")
    fig2 =bertmodel.visualize_topics()
    st.plotly_chart(fig2,use_container_width=True)
    st.header("Hierarchical Clustering")
    fig3 =bertmodel.visualize_hierarchy(top_n_topics=30)
    st.plotly_chart(fig3,use_container_width=True)
    st.header("Topic Probability Distribution")
    fig4 = bertmodel.visualize_distribution(probs, min_probability=0.015)
    st.plotly_chart(fig4,use_container_width=True)



    
if 'df' in st.session_state:
    if option == "Sentiment":
        sentiment()

if 'data' in st.session_state:
    if option =="Visual Analysis":
        st.dataframe(st.session_state.data.head(5),width=1500)
        pie_chart_data= st.session_state.data.copy()
        pie_chart(pie_chart_data)
    if option =="Words Distribution":
        cloud()

        #summarization
        concatenated_reviews = ' '.join(st.session_state.filtered_reviews["Reviews"])
        concatenated_reviews = concatenated_reviews 
        st.subheader("Summarization of selected words")
        prompt=f"""Please analyze the provided customer reviews and generate a summary that encapsulates the critical feedback 
        and areas for improvement highlighted by the customers. Focus on identifying common issues or concerns raised, 
        and present them in a constructive manner. The summary should be clear, objective, and limited to 150 words. Here are my overall review
        {concatenated_reviews}."""

        review_summary = llm.generate_content(prompt).text
        st.success(review_summary)

    if option =="Reviews Authenticity":
        aunthenticity()

    if option == "Topic Modelling":
        top_n_topics = st.text_input("Enter No of topics to perform 🔢")
        if top_n_topics:
            top_n_topics = int(top_n_topics)
            topic(top_n_topics)
        
        
