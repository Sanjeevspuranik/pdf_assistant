import os
from typing import Annotated, List, TypedDict
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(override=True)  # Load environment variables from .env file

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
api_key = os.getenv("OPENAI_API_KEY", "")


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context_docs: List[object]
    relevant_images: List[dict]


class PDFGraph:
    def __init__(self, vector_db, all_pil_images):
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        self.all_pil_images = all_pil_images
        self.llm = ChatOpenAI(model=model, api_key=api_key)

        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        self.app = workflow.compile()

    def retrieve(self, state):
        query = state["messages"][-1].content
        docs = self.retriever.invoke(query)
        pages = {d.metadata['page'] for d in docs}
        imgs = [i for i in self.all_pil_images if i['page'] in pages]
        return {"context_docs": docs, "relevant_images": imgs}

    def generate(self, state):
        context = "\n".join([d.page_content for d in state["context_docs"]])
        prompt = [
            AIMessage(content=f"Context:\n{context}")] + state["messages"]
        return {"messages": [self.llm.invoke(prompt)]}
