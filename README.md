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
