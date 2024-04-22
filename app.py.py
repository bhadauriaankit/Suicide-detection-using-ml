%%writefile app.py
import streamlit as st
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import plotly.express as px
import pandas as pd
import pickle

# Specify the correct path to tokenizer.pkl
tokenizer_path = 'tokenizer.pkl'

# Load tokenizer from file
try:
    with open(tokenizer_path, 'rb') as tokenizer_file:
        token_form = pickle.load(tokenizer_file)
except FileNotFoundError:
    print(f"Error: '{tokenizer_path}' not found.")
    # Handle the error gracefully

token_form = pickle.load(open('tokenizer.pkl', 'rb'))
model = load_model("model.h5")

if __name__ == '__main__':
    st.title('Suicidal Post Detection App ')
    st.subheader("Input the Post content below")
    sentence = st.text_input("Enter your post content here")
    predict_btt = st.button("Predict")

    if predict_btt:
        
        st.write("Post: " +sentence)
        twt = [sentence]
        twt = token_form.texts_to_sequences(twt)
        twt = pad_sequences(twt, maxlen=50)

        prediction = model.predict(twt)[0][0]
      
        if(prediction > 0.5):
             st.warning("Potential Suicide Post")
        else:
            st.success("Non Suicide Post")
        class_label = ["Potential Suicide Post","Non Suicide Post"]
        prob_list = [prediction*100,100-prediction*100]
        prob_dict = {"Potential Suicide Post/Non Suicide Post":class_label,"Probability":prob_list}
        df_prob = pd.DataFrame(prob_dict)
        fig = px.bar(df_prob, x='Potential Suicide Post/Non Suicide Post', y='Probability')
        model_option = "LSTM+GLove"
        if prediction > 0.5:
            fig.update_layout(title_text="{} model - prediction probability comparison between Potential Suicide Post and Non Suicide Post".format(model_option))
            st.info("The {} model predicts that there is a higher {} probability that the post content is Potential Suicide Post compared to a {} probability of being Non Suicide Post".format(model_option,prediction*100,100-prediction*100))
        else:
            fig.update_layout(title_text="{} model - prediction probability comparison between Potential Suicide Post and Non Suicide Post".format(model_option))
            st.info("Your post content is rather abstract, The {} model predicts that there a almost equal {} probability that the post content is Potential Suicide Post compared to a {} probability of being Non Suicide Post".format(model_option,prediction*100,100-prediction*100))
        st.plotly_chart(fig, use_container_width=True)