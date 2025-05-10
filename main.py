import streamlit as st
import requests
import os
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel
import tempfile
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

load_dotenv()

class CandidateInformation(BaseModel):
    candidate_name: str
    education: list[str]
    work_experience: list[str]
    skills: list[str]
    certifications: list[str]
    publications: list[str]
    projects: list[str]
    fit_score: str
    skills_match: str
    experience_match: str
    decision_explanation: str

class CustomMessagesState(MessagesState):
    candidate_name: str = ""
    job_description: str = ""
    resume_text: str = ""
    education: list[str] = []
    work_experience: list[str] = []
    skills: list[str] = []
    certifications: list[str] = []
    publications: list[str] = []
    projects: list[str] = []
    fit_score: str = ""
    skills_match: dict[str, float] = {}
    experience_match: dict[str, float] = {}
    decision_explanation: str = ""
    
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=10,
)
    
@tool
async def search_candidate_online(name: str, last_organization: str) -> str:
    """
    Searches for candidate information online using Google Search API and extracts content from top results.
    
    Args:
        name: The candidate's name
        last_organization: The candidate's last organization or company
        
    Returns:
        Combined content from top search results (GitHub, LinkedIn, personal websites)
    """
    try:
        query = f"{name} {last_organization}"
        st.info(f"Searching online for: {query}")
        urls = google_search(query)
        
        if not urls:
            return "No search results found for the candidate."
        
        # Extract content from each URL
        all_content = []
        for url in urls:
            try:
                st.info(f"Extracting content from: {url}")
                loader = WebBaseLoader(url)
                docs = await loader.aload()
                content = "\n\n".join([doc.page_content for doc in docs])
                all_content.append(f"Source: {url}\n{content}")
            except Exception as e:
                st.error(f"Error extracting content from {url}: {str(e)}")
                all_content.append(f"Source: {url}\nError: Could not extract content")
        
        return "\n\n===================================\n\n".join(all_content)
    except Exception as e:
        return f"Error performing online search: {str(e)}"

def google_search(query, api_key=os.getenv("GOOGLE_API_SECRET"), num_results=5):
    """
    Performs a Google search using the Custom Search API.
    
    Args:
        query: The search query
        api_key: Google API key
        num_results: Number of results to return
        
    Returns:
        List of URLs from search results
    """
    search_engine_id = os.getenv("GOOGLE_CX_SECRET")
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}&num={num_results}"

    response = requests.get(url)
    results = response.json()
    
    urls = []
    if "items" in results:
        for item in results["items"][:num_results]:
            urls.append(item.get("link"))
            
    st.info(f"Found {len(urls)} urls: {urls}")
    return urls

@tool
async def extract_url_content(url: str) -> str:
    """
    Extracts and returns the content from a given URL.
    
    Args:
        url: The URL to extract content from. Must be a valid URL starting with http:// or https://
        
    Returns:
        The extracted text content from the URL
    """
    try:
        st.info(f"Extracting content from URL: {url}")
        loader = WebBaseLoader(url)
        docs = await loader.aload()
        content = "\n\n".join([doc.page_content for doc in docs])
        return content
    except Exception as e:
        return f"Error extracting content from URL: {str(e)}"

tools = [extract_url_content, search_candidate_online]

workflow = StateGraph(CustomMessagesState)    
    
def get_user_information(state: CustomMessagesState):
    
    always_multiply_llm = llm.bind_tools(tools, tool_choice="search_candidate_online").with_structured_output(CandidateInformation)
    
    # Create a more structured prompt to help the agent complete its task more efficiently
    prompt = f"""
    Analyze the following resume and job description to extract candidate information:
    
    RESUME:
    {state.get('resume_text')}
    
    JOB DESCRIPTION:
    {state.get('job_description')}
    
    IMPORTANT: You MUST use the search_candidate_online tool to find additional information about the candidate online.
    Also use the extract_url_content tool if you find any URLs in the resume.
    This is required to provide a complete assessment.
    
    Extract all relevant information and provide a structured response.
    """
    
    # Force tool usage by adding a system message that instructs the agent to use tools
    system_message = """You are a candidate assessment assistant. 
    For every candidate, you MUST use the search_candidate_online tool to gather additional information.
    If you find any URLs in the resume, you MUST use the extract_url_content tool to analyze them.
    Do not skip using these tools as they provide critical information for your assessment."""
    
    candidate_info = always_multiply_llm.invoke(prompt)
    st.success("Analysis complete!")
    
    if candidate_info:
        state['candidate_name'] = candidate_info.candidate_name
        state['education'] = candidate_info.education
        state['work_experience'] = candidate_info.work_experience
        state['skills'] = candidate_info.skills
        state['certifications'] = candidate_info.certifications
        state['publications'] = candidate_info.publications
        state['projects'] = candidate_info.projects
        state['fit_score'] = candidate_info.fit_score
        state['skills_match'] = candidate_info.skills_match
        state['experience_match'] = candidate_info.experience_match
        state['decision_explanation'] = candidate_info.decision_explanation
    
    return state

workflow.add_node("agent", get_user_information)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

app_flow = workflow.compile()

def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        pdf_loader = PyPDFLoader(temp_file_path)
        pdf_pages = pdf_loader.load()
        file_content = "\n\n".join([page.page_content for page in pdf_pages])
        return file_content
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def display_candidate_info(candidate_details):
    st.header("Candidate Assessment Results")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        st.write(f"**Name:** {candidate_details.get('candidate_name', 'N/A')}")
        st.write(f"**Fit Score:** {candidate_details.get('fit_score', 'N/A')}")
        
        st.subheader("Education")
        for edu in candidate_details.get('education', []):
            st.write(f"- {edu}")
            
        st.subheader("Skills")
        for skill in candidate_details.get('skills', []):
            st.write(f"- {skill}")
    
    with col2:
        st.subheader("Work Experience")
        for exp in candidate_details.get('work_experience', []):
            st.write(f"- {exp}")
            
        st.subheader("Certifications")
        for cert in candidate_details.get('certifications', []):
            st.write(f"- {cert}")
            
        st.subheader("Projects")
        for proj in candidate_details.get('projects', []):
            st.write(f"- {proj}")
    
    st.subheader("Decision Explanation")
    st.write(candidate_details.get('decision_explanation', 'No explanation provided'))
    
    if candidate_details.get('publications', []):
        st.subheader("Publications")
        for pub in candidate_details.get('publications', []):
            st.write(f"- {pub}")

def main():
    st.set_page_config(page_title="Candidate Assessment Tool", layout="wide")
    
    st.title("Candidate Assessment Tool")
    st.write("Upload a resume and job description to analyze candidate fit")
    
    with st.form("candidate_form"):
        candidate_name = st.text_input("Candidate Name")
        resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
        job_description = st.text_area("Job Description", height=200)
        
        submit_button = st.form_submit_button("Analyze Candidate")
    
    if submit_button and resume_file and job_description and candidate_name:
        with st.spinner("Analyzing candidate information..."):
            try:
                # Extract text from PDF
                file_content = extract_text_from_pdf(resume_file)
                
                # Build message history
                messages = []
                messages.append(
                    AIMessage(content="Extract the information from the resume and return it in a structured format, if you find url in the resume, \
                            extract the content from the extract_url_content tool, else use search_candidate_online tool to get user related information and return it in the structured format, and give fit score from \
                            'Strong Fit', 'Moderate Fit', or 'Not a Fit' and comparision metrix and also decision_explanation and content_found_from_online"))
                
                # Prepare initial state
                initial_state = {
                    "messages": messages,
                    "resume_text": file_content,
                    "candidate_name": candidate_name,
                    "job_description": job_description
                }
                
                # Run the workflow
                candidate_details = app_flow.invoke(initial_state)
                
                # Display results
                display_candidate_info(candidate_details)
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
    
    elif submit_button:
        st.warning("Please fill in all required fields and upload a PDF resume.")

if __name__ == "__main__":
    main()
