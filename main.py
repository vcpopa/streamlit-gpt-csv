import os
import platform
import ast
import openai
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain,PromptTemplate
import pandas as pd
import streamlit as st
from streamlit_chat import message
from sqlalchemy import create_engine
from openai.error import RateLimitError
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import numpy as np
openai.api_key=st.secrets.__getitem__("OPENAI_API_KEY")
llm = OpenAI(temperature=0,model_name='text-davinci-003',max_retries=0)
def generate_plot(user_query,data):
    plot_agent=create_pandas_dataframe_agent(llm,data,verbose=True,prefix='You are a data visualization assistant that only knows python and nothing. Every one of your answers must return a seaborn plot object')
    response=plot_agent.run(user_input)
    return response


if __name__=="__main__":

    st.title("CSV Chat")

    file=r"./heart.csv"
    

    if file is not None:
    # User input
        df=pd.read_csv(file)
        user_input = st.text_input("Ask a question", value="", key="user_input")
        submit=st.button("Submit")
        if user_input!='' and submit:
        # Call custom function to generate response
            response = generate_plot(user_input,data=df)

            st.pyplot(exec(f"""
import seaborn as sns
import matplotlib.pyplot as plt
{response}
"""))
        
        if st.button("Exit"):
            st.write("Chatbot: Goodbye!")
