# LangSmith Agent & Tool Integration in LangChain with Python
This project demonstrates a workflow for document retrieval and question-answering with LangSmith, LangChain, and various tools (Wikipedia, Arxiv, custom retrievers). It leverages Google Colab for setup and deployment.

## Prerequisites
1. Python 3.10+
2. Google Colab or similar notebook environment
3. API Keys for required services:
4. GROQ API Key (for llama3-70b-8192 model)
### Installation
**Run the following commands in Google Colab to install dependencies:**
```bash
!pip install langchain-groq
!pip install arxiv
!pip install langchain_community
!pip install "unstructured[md]"
!pip install chromadb
!pip install langchain
!pip install wikipedia
!pip install sentence-transformers
!pip install langchainhub
```

### Environment Setup
**Ensure the GROQ API key is set up in your environment:**
```bash
import os
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
```
## Tools and Agents
#### Wikipedia Tool
**Initialize the Wikipedia tool to retrieve summaries for a given topic.**

```bash
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api)
```

#### Retriever Tool
1. **Document Loading:** Use WebBaseLoader to load data from specified web pages.
```bash
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
```
2. **mbedding Setup:** Set up HuggingFace embeddings to vectorize the documents for retrieval.
```bash
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
```

3. **Vector Database:** Use Chroma as the vector database for efficient document retrieval.

```bash
from langchain.vectorstores import Chroma

vectordb = Chroma.from_documents(documents, embeddings)
retriever = vectordb.as_retriever()
```

4. **Create Retriever Tool:** Configure the retriever tool for targeted searches.
```bash
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(retriever, "langsmith_search", 
                                       "Search for information about LangSmith.")
```

### Arxiv Tool
**To query research papers from Arxiv:**
```bash
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=100)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
```

### Agent Setup
Integrate the above tools into a LangSmith agent to streamline queries.
```bash
tools = [wiki, arxiv, retriever_tool]

from langchain_groq import ChatGroq
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### Example Queries
**Run queries with agent_executor:**
```bash
# Example query about LangSmith
agent_executor.invoke({"input": "Tell me about Langsmith in big paragraph form"})

# Example query about a specific paper from Arxiv
agent_executor.invoke({"input": "What's the paper 1605.08386 about?"})
```


