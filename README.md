# Crear una aplicación de generación aumentada (RAG) de recuperación: parte 1 y 2

## Desarrollado por:
* Andrés Felipe Rodríguez Chaparro

## Arquitectura y componentes
La arquitectura de la aplicación RAG  consta de los siguientes componentes:​

* Fuente de Datos: La información sobre la cual se realizarán las consultas. Puede ser una base de datos, documentos de texto, páginas web, etc.​
* Indexación de Datos: Proceso de convertir la fuente de datos en un formato que permita una recuperación eficiente. Esto puede incluir la tokenización y la creación de embeddings almacenados en una base de datos vectorial.​
* Módulo de Recuperación: Al recibir una consulta, este módulo busca en la base de datos vectorial para encontrar fragmentos de información relevantes que coincidan semánticamente con la consulta, en nuestro caso es Pinecone 
* Modelo de Lenguaje (LLM): Utiliza la información recuperada para generar una respuesta coherente y contextualizada a la consulta del usuario, en nuestro caso usaremos OpenAi para gebnerar embeddings

## Instalación

1) Actualizar el pip

```
pip install --upgrade pip
```

2) Instalamos las dependencias necesarias

```
%pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph
!pip install -U langchain langchain-openai
!pip install -qU langchain-core
!pip install -qU langchain-pinecone
```

3) Configuración
Configuramos una API KEY de OpenAi suministrada por el profesor

```
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
```
```
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```
4) Configuración del indice de Pinecone

Usando la biblioteca Pinecone para trabajar con una base de datos vectorial, creamos un índice si no existe y luego acceddemos a él.

```
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

index_name = "quixkstart"

pc = Pinecone(api_key="pcsk_3gkekU_466Jxsjdv7CwHry81ScExx4g4uP8DEVVg2w5L5QWLRfi7mj6cgsXThfAUCcPLyU")


pc.create_index(
    name=index_name,
    dimension=3072,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```

En mi caso, cree 3

```
[{
    "name": "quickstart",
    "dimension": 3072,
    "metric": "cosine",
    "host": "quickstart-o5fku1s.svc.aped-4627-b74a.pinecone.io",
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    },
    "status": {
        "ready": true,
        "state": "Ready"
    },
    "deletion_protection": "disabled"
}, {
    "name": "quixkstart",
    "dimension": 3072,
    "metric": "cosine",
    "host": "quixkstart-o5fku1s.svc.aped-4627-b74a.pinecone.io",
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    },
    "status": {
        "ready": true,
        "state": "Ready"
    },
    "deletion_protection": "disabled"
}, {
    "name": "andresarep",
    "dimension": 3072,
    "metric": "cosine",
    "host": "andresarep-o5fku1s.svc.aped-4627-b74a.pinecone.io",
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    },
    "status": {
        "ready": true,
        "state": "Ready"
    },
    "deletion_protection": "disabled"
}]
```
5) Indexación de documentos en al base de datos

Básicamente, toma contenido web, lo divide en fragmentos, lo indexa en una base de datos vectorial y luego lo usa para responder preguntas basándose en la información recuperada, usando con LangChain y LangGraph

```
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
```

## Uso y evidencias 

A partir de aqui, podemos hacerle preguntas 

```
result = graph.invoke({"question": "What is Task Decomposition?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')
```

Respuesta

Context: [Document(id='a4cd7b24-264b-4e37-ba69-484e37718861', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}, page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.'), Document(id='6ea44784-afe9-4e06-ac28-4c847f0b49a0', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}, page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.'), Document(id='8ca69984-3d35-41f9-b7e2-142cec16a27f', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}, page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.'), Document(id='b91d088f-0b2a-4294-9ead-1ec236cf727a', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}, page_content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.')]


Answer: Task Decomposition is the process of breaking down complex tasks into smaller, more manageable steps, often using techniques like Chain of Thought (CoT) prompting. This method enhances model performance by instructing the model to "think step by step," making it easier to tackle hard tasks. Additionally, approaches like Tree of Thoughts extend this concept by exploring multiple reasoning possibilities for each step.


en Pinecone podemos ver la grabación 

![Captura de pantalla 2025-03-21 175018](https://github.com/user-attachments/assets/68654b15-b630-4798-aed6-7f67bbbda70e)


## Crear documentos

sumado a eso, podemos crear documentos y cargarlos en el pinecone para que podamos preguntarle

```
from uuid import uuid4
from langchain_core.documents import Document

documents = [
    Document(page_content="I ran 5 kilometers today and felt amazing afterwards!", metadata={"source": "tweet"}),
    Document(page_content="Scientists have discovered a new exoplanet that may support life.", metadata={"source": "news"}),
    Document(page_content="Working on a deep learning project—can't wait to share my results!", metadata={"source": "tweet"}),
    Document(page_content="A major cyberattack has compromised millions of user accounts.", metadata={"source": "news"}),
    Document(page_content="Just finished reading an incredible book—highly recommend it!", metadata={"source": "tweet"}),
    Document(page_content="This year's best smartphones ranked from worst to best.", metadata={"source": "website"}),
    Document(page_content="Top 5 strategies for improving your personal finances.", metadata={"source": "website"}),
    Document(page_content="LangChain is revolutionizing AI applications with its framework.", metadata={"source": "tweet"}),
    Document(page_content="The stock market surged 300 points today after positive economic news.", metadata={"source": "news"}),
    Document(page_content="Feeling nervous about my upcoming job interview!", metadata={"source": "tweet"}),
]

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```

y si le hacemos las siguiente consulta

```
query = "Have scientists found a new planet similar to Earth?"

results = vector_store.similarity_search(query, k=2)  

print("Resultados de la consulta semántica:")
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

```
Nos dara la siguiente respuesta 


Resultados de la consulta semántica:
* Scientists have discovered a new exoplanet that may support life. [{'source': 'news'}]
* FAISS (Facebook AI Similarity Search): It operates on the assumption that in high dimensional space, distances between nodes follow a Gaussian distribution and thus there should exist clustering of data points. FAISS applies vector quantization by partitioning the vector space into clusters and then refining the quantization within clusters. Search first looks for cluster candidates with coarse quantization and then further looks into each cluster with finer quantization.
ScaNN (Scalable Nearest Neighbors): The main innovation in ScaNN is anisotropic vector quantization. It quantizes a data point $x_i$ to $\tilde{x}_i$ such that the inner product $\langle q, x_i \rangle$ is as similar to the original distance of $\angle q, \tilde{x}_i$ as possible, instead of picking the closet quantization centroid points. [{'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}]
