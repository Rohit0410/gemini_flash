from dotenv import load_dotenv

load_dotenv()
import base64
import streamlit as st
import os
import io
from PIL import Image 
import google.generativeai as genai
import pandas as pd
import docx2txt
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
import re
from llama_index.core import SimpleDirectoryReader
import random
api_list = ["AIzaSyA51WTz0t69sBFs8D2ZmLLypKs6X9rIcEI","AIzaSyDlCk6V9XXwHEYJSjSC4-g28N69UgNcVYA"]
api_key=random.choices(api_list)
print(api_key[0])
import google.generativeai as genai
genai.configure(api_key=api_key[0])


def preprocessing(document):
    text = document.replace('\n', ' ').replace('\t', ' ').lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]
    tokens = [token for token in tokens if token and token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def get_gemini_response(input,pdf_cotent,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([input,pdf_content,prompt],generation_config = genai.GenerationConfig(
        temperature=0.7
    ))
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        data = SimpleDirectoryReader(input_files=[uploaded_file]).load_data()
        document_resume = " ".join([doc.text.replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ') for doc in data])
        final = preprocessing(document_resume)
        return final
    else:
        raise FileNotFoundError("No file uploaded")
    

## Streamlit App

st.set_page_config(page_title="ATS Resume EXpert")
st.header("ATS Tracking System")
input_text=st.text_area("Job Description: ",key="input")
# jd_upload=st.file_uploader("Upload your job description...")
# if jd_upload is not None:
#     input_text = input_pdf_setup(jd_upload)
# else:
#     st.write("Please uplaod the JD")
uploaded_file=st.file_uploader("Upload your resume...",accept_multiple_files=True)
print("uploaded files",uploaded_file)


if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")


submit1 = st.button("Tell Me About the Resume")

#submit2 = st.button("How Can I Improvise my Skills")

submit3 = st.button("Percentage match")

input_prompt4="""your are an skilled Topic modelling model. get me the job title mentioned in the job description.
the output should be in this way, position,industry,sub-industry  and don't give the explaination follow the example provided. e.g. AI-Engineer, IT sector, Product based. """

model=genai.GenerativeModel('gemini-1.5-flash')
response1=model.generate_content([input_text,input_prompt4],generation_config = genai.GenerationConfig(
    temperature=0.3))
print('response1',response1.text)

input_prompt1 = """
 You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
  Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of {} and  ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as there is any gap in the career and matching percentage and can we consider him C-suite or C-1 professional e.g. Yes(reason) or No(reason) and then keywords missing and last final thoughts.
""".format(response1.text)

print("input",input_prompt3)

if submit1:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt1,pdf_content,input_text)
        st.subheader("The Repsonse is")
        st.write(response)
    else:
        st.write("Please uplaod the resume")


elif submit3:
    Dict = {}
    resume_embeddings = []  # Dictionary to store resume embeddings
    not_uploaded=[]
    RESUME_folder_path = r'D:/Rohit/jdcv_score_app/jdcv_score_app/temp4'
    os.makedirs(RESUME_folder_path,exist_ok=True)
    print(f"The directory {RESUME_folder_path} has been created successfully.")

    st.header(response1.text)

    if uploaded_file is not None:
        for i in uploaded_file:
            print('kkkkkkk',len(uploaded_file))
            print('done')
            if i is not None:
                try:
                    file_path = os.path.join(RESUME_folder_path, i.name)
                    print('FILE',file_path)
                    with open(file_path, "wb") as f:   
                        f.write(i.getbuffer()) 
                        resume_embeddings.append(file_path)
                        print('ttt',resume_embeddings)
                except Exception as e:
                    print('hh')
        for i in resume_embeddings:
            pdf_content = input_pdf_setup(i)
            print('pdf',pdf_content)
            response=get_gemini_response(input_prompt3,pdf_content,input_text)
            st.subheader(os.path.basename(i))
            # st.write(str(i))
            st.write(response)
    else:
        st.write("Please upload the resume")
