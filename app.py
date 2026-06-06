from fastapi import FastAPI
from pydantic import BaseModel
from agent_code import workflow
from fastapi.exceptions import HTTPException
from langchain_core.messages import HumanMessage

class APIagent(BaseModel):
    message:str
    thread_id:str="default_thread"

app=FastAPI(
    title="Enterprise Customer Support AI Engine",
    description="Production API wrapper for LangGraph RAG + SQL State Machine",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "online", "engine": "LangGraph-FastAPI-v1"}

@app.post("/api/v1/chat")
def Chat(payload:APIagent):
    thread=payload.thread_id
    message=payload.message
    try:
        response=workflow.invoke({"messages":[HumanMessage(content=message)]},config={"configurable":{"thread_id":thread}})
        return {"status":"sucess","thread_id":thread,"AImessage":response["messages"][-1].content,"summary":response.get("summary","")}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))