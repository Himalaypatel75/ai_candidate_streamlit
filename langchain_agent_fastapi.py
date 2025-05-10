from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import requests
import uvicorn
from pydantic import BaseModel
import os
from datetime import datetime
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from typing import Optional

load_dotenv()

app = FastAPI(title="Candidate Information Extractor API")


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=10,
)

    
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        print(f"Searching online for: {query}")
        urls = google_search(query)
        
        if not urls:
            return "No search results found for the candidate."
        
        # Extract content from each URL
        all_content = []
        for url in urls:
            try:
                print(f"Extracting content from: {url}")
                loader = WebBaseLoader(url)
                docs = await loader.aload()
                content = "\n\n".join([doc.page_content for doc in docs])
                all_content.append(f"Source: {url}\n{content}")
            except Exception as e:
                print(f"Error extracting content from {url}: {str(e)}")
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
            
    print(f"Found {len(urls)} urls: {urls}")
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
        print(f"Extracting content from URL::::::::::::::::::::::::::::: {url}")
        loader = WebBaseLoader(url)
        docs = await loader.aload()
        content = "\n\n".join([doc.page_content for doc in docs])
        return content
    except Exception as e:
        return f"Error extracting content from URL: {str(e)}"

tools = [extract_url_content , search_candidate_online]

workflow = StateGraph(CustomMessagesState)    
    
def get_user_information(state: CustomMessagesState):
    
    always_multiply_llm = llm.bind_tools(tools, tool_choice="search_candidate_online").with_structured_output(CandidateInformation)
    
    
    # agent_executor = create_react_agent(llm, tools, checkpointer=MemorySaver(), response_format=CandidateInformation)
    # # Set a higher recursion limit to prevent GraphRecursionError
    # config = {"recursion_limit": 50}
    
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
    print(f"result: {candidate_info}")
    # result = agent_executor.invoke(
    #     {"messages": [HumanMessage(content=system_message), HumanMessage(content=prompt)]},
    #     config=config
    # )

    # Extract candidate information from result and set to state
    # candidate_info = result.get("structured_response", None)
    # print(f"candidate_info: {candidate_info} | type: {type(candidate_info)}")
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
        state['content_found_from_online'] = candidate_info.content_found_from_online
    
    return state

workflow.add_node("agent", get_user_information)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

app_flow = workflow.compile()

# File upload and processing endpoint
@app.post("/extract_candidate_info")
async def extract_candidate_info(
    candidate_name: str = Form(...),
    resume_file: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        # Validate file type
        if not resume_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = f"temp/{timestamp}_{resume_file.filename}"
        # Save the file to disk
        with open(file_path, "wb") as f:
            f.write(await resume_file.read())
            
        # Import PDF extraction library
        from langchain_community.document_loaders import PyPDFLoader
        
        # Extract text from the PDF
        pdf_loader = PyPDFLoader(file_path)
        pdf_pages = pdf_loader.load()
        file_content = "\n\n".join([page.page_content for page in pdf_pages])


        # Build message history from request.history
        messages = []
        messages.append(
            AIMessage(content="Extract the information from the resume and return it in a structured format, if you find url in the resume, \
                      extract the content from the extract_url_content tool,else use search_candidate_online tool  to get user related information and return it in the structured format, and give fit score from \
                          'Strong Fit', 'Moderate Fit', or 'Not a Fit' and comparision metrix and also decision_explanation and content_found_from_online"))
        

        # print(f"file_content: {file_content}")
        initial_state = {
            "messages": messages,
           "resume_text" : file_content,
           "candidate_name" : candidate_name,
           "job_description" : job_description
        }

        
        candidate_details = app_flow.invoke(initial_state)
        return candidate_details
        
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    finally:
        # Clean up - remove the temporary file
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)





def main():
    uvicorn.run("langchain_agent:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()

