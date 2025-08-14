import streamlit as st
import PyPDF2
import docx2txt
import io
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="AI Career Predictor", layout="centered")
st.title("AI Career Predictor")
st.markdown("Upload your resume to get the top 3 most suitable job roles predicted by an AI model.")

uploaded_file = st.file_uploader("Upload Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

def extract_text_from_resume(file_bytes, filename):
    if filename.endswith(".pdf"):
        text = ""
        file_stream = io.BytesIO(file_bytes)
        reader = PyPDF2.PdfReader(file_stream)
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif filename.endswith(".docx"):
        file_stream = io.BytesIO(file_bytes)
        return docx2txt.process(file_stream)
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8")
    else:
        raise ValueError("Unsupported file type. Upload .pdf, .docx, or .txt")

@st.cache_resource
def load_model():
    from huggingface_hub import login
    login(token="YOUR_HF_TOKEN_HERE")  # replace with your real token
    
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

if uploaded_file:
    filename = uploaded_file.name
    file_bytes = uploaded_file.read()
    resume_text = extract_text_from_resume(file_bytes, filename)

    if st.button("Predict"):
        with st.spinner("Analyzing your resume..."):
            llm = load_model()

            template = """You are a career counselor. Given the following resume text, extract key skills, experience, and education. 
Then predict the top 3 job roles the candidate is suited for, with a brief explanation for each.

Output format must be:
1. Job Role - 1-line reason
2. Job Role - 1-line reason
3. Job Role - 1-line reason

Now, analyze this resume and provide predictions.

Resume:
{text}

Prediction: """

            prompt = PromptTemplate(input_variables=["text"], template=template)
            chain = LLMChain(llm=llm, prompt=prompt)

            prediction = chain.run(resume_text)
            st.success("Prediction complete!")
            st.markdown("### Top 3 Predicted Job Roles:")
            st.markdown(prediction)

