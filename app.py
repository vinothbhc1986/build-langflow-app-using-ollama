# Builds the Langraph workflow
#olllama--> runs LLM locally on your laptop, no cloud API needed

# package: 
  #langchain-ollama -->provides the 'char-ollama' module for interacting with the Ollama LLM
  # langgraph --> provides the 'graph' module for building and managing state graphs



from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict

llm = ChatOllama(model="llama3.2:3b")

# Define the state structure for the workflow. Typed dictionary that holds
  #i) messages: the input conversation messages to the LLM
  #ii) response: the output response from the LLM (model's generated text)

class State(TypedDict):
    messages: list
    response: str

# this is 'single graph' node that takes the input messages, invokes the LLM, and returns the generated response
def generate(state: State) -> State:
    messages = state["messages"] #  reads the input messages from the currentstate
    response = llm.invoke(messages) # sends them to the local LLM using llm.invoke()
    return {"response": response.content} # returns the model's text response which gets merged into the state

# Builds the Langraph workflow
graph = StateGraph(State) # Creats a 'StateGraph' with the defined state schema
graph.add_node("generate", generate) # Regsister the 'generate' function as a node in the graph
graph.set_entry_point("generate") # Sets the 'generate' as entry point for the graph. 
graph.add_edge("generate", END) # Adds an edge from the 'generate' node to the END node, indicating that after generating the response, the workflow ends.

app = graph.compile() #  compile the grpah into an executable workflow that can be invoked with an initial state.

initial_state = {"messages": [("user", "Why is AI the future")]}
result = app.invoke(initial_state)
print(result["response"])