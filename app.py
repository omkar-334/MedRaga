import asyncio
import json
import logging
import os
import random
import time

import cohere
import requests
from dotenv import dotenv_values
from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore

from mthrottle import Throttle
from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from qdrant_client import QdrantClient
from qdrant_client.models import Batch, Distance, VectorParams
from serpapi import GoogleSearch

logger = logging.getLogger("PyPDF2")
logger.setLevel(logging.ERROR)

throttle_config = {"lookup": {"rps": 15}, "default": {"rps": 8}}
th = Throttle(throttle_config, 15)

apidict = dict(dotenv_values())

llm = Groq(model="llama3-70b-8192", api_key="your_api_key")


async def get_links(condition, n):
    queries = [i.format(condition) for i in ["{} causes", "{} risk factors", "{} symptoms", "{} prevention and cure", "{} diagnosis", "{} treatment options", "{} prognosis", "{} complications", "{} epidemiology", "{} research and studies", "{} latest treatments and advancements"]]

    output = []
    for query in queries:
        params = {"q": query, "num": n, "hl": "en", "source": "python", "serp_api_key": apidict["SERPAPI_KEY"]}
        search = GoogleSearch(params).get_dict()
        links = [i["link"] for i in search["organic_results"]]
        output.extend(links)
    return output


def download_all(links, path):
    links = list(set(links))
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/118.0", "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.5"})

    idx = 0
    for filelink in links:
        idx += 1
        filename = f"{path}\\file{idx}.pdf"
        download(session, filelink, filename)
    session.close()
    return len(links)


def download(session, url, filename):
    th.check()
    try:
        with session.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(filename, mode="wb") as f:
                for chunk in r.iter_content(chunk_size=1000000):
                    f.write(chunk)
    except:
        return False


def valid_pdf(file):
    try:
        pdf = PdfReader(file)
        return True
    except PdfReadError:
        print(f"{file} - Invalid file")
        return False


def clean_dir(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filepath.lower().endswith(".pdf"):
            if not valid_pdf(filepath):
                print(f"invalid file - {filepath}")
                os.remove(filepath)
    print("Directory cleaned")

def create_collection(name):
    client = QdrantClient(url="http://localhost:6333")
    client.create_collection(
        collection_name="MedicalPapers",
        vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )

def embed(textlist):
    cohere_client = cohere.Client(COHERE_API_KEY)
    cohere_client.embed(
                model="embed-english-v3.0",
                input_type="search_document",
                texts=textlist,
            )

def embed_docs(path):
    reader = SimpleDirectoryReader(input_dir=path)
    documents = reader.load_data()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100, length_function=len, is_separator_regex=False)
    texts = text_splitter.create_documents([pages[i].page_content for i in range(len(pages))])
    # qdrant = Qdrant.from_documents(docs, embeddings, url=QDRANT_URL, collection_name=QDRANT_CLUSTER, api_key=QDRANT_KEY, force_recreate=True)
    client = QdrantClient()
    client.upsert(
        collection_name="MedicalPapers",
        points=Batch(
            ids=range(len(texts)),
            vectors=.embeddings,
            payloads=[{"Context{}".format(index): value} for index, value in enumerate([texts[i].page_content for i in range(len(texts))], start=1)],
        ),
    )


def cohereRetrival(collection, textList):
    cohere_client = cohere.Client(COHERE_API_KEY)
    client = QdrantClient()
    result = client.search(
        collection_name=collection,
        query_vector=cohere_client.embed(
            model="embed-english-v3.0",
            input_type="search_query",
            texts=textList,
        ).embeddings[0],
    )
    return result


def ragFusion(prompt, collection="MedicalPapers"):
    co = cohere.Client(COHERE_API_KEY)
    queryGenerationPrompt = ChatPromptTemplate.from_template("Given the prompt: '{prompt}', generate {num_queries} questions that are better articulated. Return in the form of an list. For example: ['question 1', 'question 2', 'question 3']")
    queryGenerationChain = queryGenerationPrompt | llm1
    queries = queryGenerationChain.invoke({"prompt": prompt, "num_queries": 3}).content.split("\n")
    retrievedContent = []
    for query in queries:
        ret = cohereRetrival(collection, [query])
        for doc in ret:
            for key, value in doc.payload.items():
                value = value.replace("\xa0", " ").replace("\t", "  ").replace("\r", "").replace("\n", "      ")
                retrievedContent.append(value)
    retrievedContent = list(set(retrievedContent))
    result = co.rerank(model="rerank-english-v3.0", query=prompt, documents=retrievedContent, top_n=5, return_documents=True)
    context = ""
    for i in result.results:
        context += i.document.text + "\n\n"
    return context


app = FastAPI()
origins = ["http://localhost.tiangolo.com", "https://localhost.tiangolo.com", "http://localhost", "http://localhost:8080", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CParams(BaseModel):
    req: str
    # n: int = 5


class QParams(BaseModel):
    req: str
    # id: int
    # prompt: str


@app.post("/create")
async def create(params=Depends(CParams)):
    start = time.time()
    req = json.loads(params.req)

    n = 4
    id = req["id"]
    condition = req["condition"]
    newpath = f".\\files\\{id}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    with open(newpath + "\\details.json", "w") as file:
        file.write(json.dumps(req))

    with open(f".\\files\\{id}\\history.txt", "w") as file:
        file.write("")
    dstring = "\n".join([f"{key}:{value}" for key, value in req.items()])

    with open(newpath + "\\details.txt", "w") as file:
        file.write(dstring)

    links = await get_google(condition, newpath, n)
    length = download_all(links, newpath)
    clean_dir(newpath)
    embed_docs(newpath)

    return {"id": id, "status": f"Downloaded {length} files", "Path": newpath, "Time taken": time.time() - start}


@app.get("/query")
async def query(params=Depends(QParams)):
    treatment_box = """You are a medical assistant that specializes in providing second opinions, diagnosing complex cases and suggesting treatment plans. When I describe the patient details, medical context and task, give me the appropriate treatment plan based on the task given by analyzing the patient details and medical context. Include how your answer is related to the patient's history. Do not print the analysis or summary of the patient's details."""
    answer_box = """
As a medical assistant specializing in second opinions, treatment plans and medical diagnoses, accurate and relevant response to the given question. Ensure the response is detailed, factually correct, coherent, and clear to understand. Answer in a factual and relevant manner, describing each step.
"""

    template = """{box}
    
    {history}

    Patient History : {details}

    Medical Context : {context}

    Task: {question}
    """
    req = json.loads(params.req)
    id = req["id"]
    userprompt = req["prompt"]
    path = f".\\files\\{id}\\details.txt"
    with open(path, "r") as file:
        details = file.readlines()
    hpath = f".\\files\\{id}\\history.txt"

    file_path = f".\\files\\{id}\\details.json"
    with open(file_path, "r") as file:
        patient_data = json.load(file)
    # Extract condition and description
    condition = patient_data["condition"]
    description = patient_data["description"]

    with open(hpath, "r") as history_file:
        history = history_file.read()
    history = f"""This is your previous chat history with your patient. Your answer should be a continuation of the conversation between you and the patient.
    Chat history : +\n{history}"""

    context = ragFusion(userprompt)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm15
    if "treatment" in prompt:
        box = treatment_box
    else:
        box = answer_box

    result = chain.invoke({"context": context[0], "details": details, "question": prompt, "box": box, "history": history})
    with open(hpath, "a") as history_file:
        history_file.write("##### Human: " + userprompt + "\n\n")
        history_file.write("##### Bot: " + result.content + "\n\n")
    return {"Output": result.content}


@app.get("/status")
def status():
    return {"status": "200 OK"}
