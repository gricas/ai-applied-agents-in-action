import json
import os
from typing import Dict, List, Any, Optional
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status, HTTPException, Body, UploadFile, File
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai.foundation_models import ModelInference, Model
from ibm_watsonx_ai import Credentials


from langchain_ibm import WatsonxLLM

# from langchain.llms.base import LLM
from crewai import Agent, Task, Crew, Process, LLM


from pydantic import BaseModel

import tempfile
import chromadb

from routes.models import ModelRequest

from schemas import (
    TestRequest,
    ExamplesTemplate,
    PromptTemplateRequest,
    JSONResponseTemplate,
    GeneratePetNameResponse,
    GenerateSummaryResponse,
    ClasificationResponse
)

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Global dictionaries to store models and prompt templates
models: Dict[str, ModelRequest] = {}
prompt_templates: Dict[str, PromptTemplateRequest] = {}
examples_templates: Dict[str, ExamplesTemplate] = {}

# Directory paths for models and prompt templates
MODELS_DIR_PATH = "data/models"
TEMPLATES_DIR_PATH = "data/prompt_templates"
EXAMPLES_DIR_PATH = "data/examples"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle the lifespan of the FastAPI application.
    Loads models and prompt templates from files at startup and
    clears them at shutdown.
    """
    try:
        load_models()
        load_prompt_templates()
        load_examples()
        yield
    finally:
        models.clear()
        prompt_templates.clear()
        examples_templates.clear()


def load_models():
    """
    Load models from JSON files.
    """
    for filename in os.listdir(MODELS_DIR_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(MODELS_DIR_PATH, filename)
            with open(file_path, "r") as f:
                model_data = json.load(f)
            template_model = filename[:-5]
            model_request = ModelRequest.from_json(model_data)
            models[template_model] = model_request


def load_prompt_templates():
    """
    Load prompt templates from text files.
    """
    for filename in os.listdir(TEMPLATES_DIR_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(TEMPLATES_DIR_PATH, filename)
            with open(file_path, "r") as f:
                template = f.read()
            template_name = filename[:-4]
            prompt_templates[template_name] = PromptTemplateRequest(template=template)


def load_examples():
    """
    Load examples from text files.
    """
    for filename in os.listdir(EXAMPLES_DIR_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(EXAMPLES_DIR_PATH, filename)
            with open(file_path, "r") as f:
                template = f.read()
            template_name = filename[:-4]
            examples_templates[template_name] = ExamplesTemplate(template=template)


# Initialize FastAPI app with custom lifespan
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """
    Health check endpoint to see if the API is up and running.
    """
    return {"message": "Fast API up!"}


@app.post("/test_route")
async def test_route(request: TestRequest):
    """
    Test route to ensure the server can receive and return data properly.
    """
    try:
        data = request.data
        return {"data": data}
    except Exception as e:
        logging.error(f"Error in test_route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


chroma_client = chromadb.PersistentClient(path="./chroma_db")


@app.post("/process-documents")
async def process_documents(files: List[UploadFile] = File(...)):
    try:

        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")

        credentials = Credentials(
            url=url,
            api_key=apikey,
        )

        embedding_model = Embeddings(
            model_id="intfloat/multilingual-e5-large",
            credentials=credentials,
            project_id=project_id,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            documents = []

            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                file_name = os.path.splitext(file.filename)[0]
                print(f"FILE NAME IS: {file_name}")
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                if file.filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.filename.endswith(".txt"):
                    loader = TextLoader(file_path)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type")

                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=500,
                length_function=len,
                is_separator_regex=False,
            )
            splits = text_splitter.split_documents(documents)

            collection = chroma_client.get_or_create_collection(
                name="document_collection", metadata={"hnsw:space": "cosine"}
            )

            batch_size = 100
            for i in range(0, len(splits), batch_size):
                batch = splits[i : i + batch_size]

                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]

                embeddings = embedding_model.embed_documents(
                    texts=texts,
                    concurrency_limit=10,
                    # params={
                    #     EmbedTextParamsMetaNames.BATCH_SIZE: min(
                    #         batch_size, len(batch))
                    # }
                )

                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=[f"doc_{i}_{j}" for j in range(len(batch))],
                )

        return {"message": f"Successfully processed {len(files)} documents"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-documents-by-collection")
async def process_documents_by_collection():
    txt_files = []
    try:
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")
        credentials = Credentials(
            url=url,
            api_key=apikey,
        )
        embedding_model = Embeddings(
            model_id="intfloat/multilingual-e5-large",
            credentials=credentials,
            project_id=project_id,
        )

        for filename in os.listdir("docs"):
            if filename.endswith(".txt"):
                file_path = os.path.join("docs", filename)
                with open(file_path, "rb") as f:
                    content = f.read()
                    txt_files.append(("files", (filename, content, "text/plain")))

        with tempfile.TemporaryDirectory() as temp_dir:
            for file_tuple in txt_files:
                _, (filename, content, _) = file_tuple
                file_path = os.path.join(temp_dir, filename)
                file_name = os.path.splitext(filename)[0]

                # Write content to temporary file
                with open(file_path, "wb") as buffer:
                    buffer.write(content)

                # Only process txt files since that's all we're collecting
                loader = TextLoader(file_path)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=500,
                    length_function=len,
                    is_separator_regex=False,
                )
                documents = loader.load()
                splits = text_splitter.split_documents(documents)
                collection = chroma_client.get_or_create_collection(
                    name=file_name,
                    metadata={"hnsw:space": "cosine"},
                )

                batch_size = 100
                for i in range(0, len(splits), batch_size):
                    batch = splits[i : i + batch_size]
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    embeddings = embedding_model.embed_documents(
                        texts=texts, concurrency_limit=5
                    )
                    collection.add(
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas,
                        ids=[f"{file_name}_{i}_{j}" for j in range(len(batch))],
                    )

        return {
            "message": f"Successfully processed {len(txt_files)} files into their respective collections"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-collections")
async def list_collections():
    """
    Route to list all collections in your chromadb.
    """
    try:
        # Get collection names
        collection_names = chroma_client.list_collections()

        # Get details for each collection
        collections_info = []
        for name in collection_names:
            collection = chroma_client.get_collection(name)
            collections_info.append({"name": name, "count": collection.count()})

        return {
            "total_collections": len(collections_info),
            "collections": collections_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear-db")
async def clear_database():
    """
    Route to clear all collections from your chromadb.
    """
    try:
        collection_names = chroma_client.list_collections()

        for name in collection_names:
            chroma_client.delete_collection(name=name)

        return {"message": f"Successfully deleted {len(collection_names)} collections"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_documents(query: str):
    """
    Simple search against one collection.
    """
    try:
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")
        credentials = Credentials(
            url=url,
            api_key=apikey,
        )

        embedding_model = Embeddings(
            model_id="intfloat/multilingual-e5-large",
            credentials=credentials,
            project_id=project_id,
        )

        collection = chroma_client.get_collection(name="document_collection")

        query_embedding = embedding_model.embed_documents(texts=[query])

        results = collection.query(query_embeddings=query_embedding, n_results=3)

        return {
            "query": query,
            "matches": {
                "documents": results["documents"][0],
                "distances": results["distances"][0],
                "ids": results["ids"][0],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search-all")
async def search_all_documents(query: str):
    """
    Search across all collections. Not best practice, but you know, just to
    show a before and after agentic RAG.
    """
    try:
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")
        credentials = Credentials(
            url=url,
            api_key=apikey,
        )
        embedding_model = Embeddings(
            model_id="intfloat/multilingual-e5-large",
            credentials=credentials,
            project_id=project_id,
        )

        query_embedding = embedding_model.embed_documents(texts=[query])

        collections = chroma_client.list_collections()

        all_results = []
        for collection_name in collections:
            collection = chroma_client.get_collection(name=collection_name)
            results = collection.query(query_embeddings=query_embedding, n_results=1)

            result = {
                "collection_name": collection_name,
                "documents": results["documents"][0],
                "distances": results["distances"][0],
                "ids": results["ids"][0],
            }
            all_results.append(result)

        sorted_results = sorted(all_results, key=lambda x: x["distances"][0])

        return {"query": query, "matches": sorted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    query: str


@app.post("/rag-query")
async def rag_query(request: QueryRequest):
    try:
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")

        credentials = Credentials(
            url=url,
            api_key=apikey,
        )

        embedding_model = Embeddings(
            model_id="intfloat/multilingual-e5-large",
            credentials=credentials,
            project_id=project_id,
            verify=True,
        )

        parameters = {
            "decoding_method": "greedy",
            "min_new_tokens": 1,
            "max_new_tokens": 300,
            "stop_sequences": ["<|endoftext|>"],
        }

        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            params=parameters,
            project_id=project_id,
        )

        watsonx_llm = WatsonxLLM(watsonx_model=model)

        query_embedding = embedding_model.embed_query(request.query)

        collection = chroma_client.get_collection("Coffee")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,  # 3 results seems like it gives good answers
            include=["documents", "metadatas", "distances"],
        )

        relevant_documents = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            metadata["collection"] = "Coffee"
            metadata["relevance_score"] = 1 - distance
            relevant_documents.append({"content": doc, "metadata": metadata})

        # so this is where we grab the response from the vectordb, and add it to a
        # context var
        context = "\n\n".join(
            [f"Content:\n{doc['content']}" for doc in relevant_documents]
        )

        # got this prompt directly from IBM's internal RAG template
        prompt = f"""<|start_of_role|>system<|end_of_role|>
                - You are a helpful, respectful, and honest assistant that can summarize long documents.
                - Always respond as helpfully as possible, while being safe.
                - Your responses should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
                - Please ensure that your responses are socially unbiased and positive in nature.
                - If a document does not make any sense, or is not factually coherent, explain why instead of responding something not correct.
                - If you don't know the response to a query, please do not share false information.
                <|start_of_role|>user<|end_of_role|>
                You are an assistant for question-answering tasks. Generate a conversational response for the given question based on the given set of document context. Think step by step to answer in a crisp manner. Answer should not be more than 200 words. If you do not find any relevant answer in the given documents, please state you do not have an answer. Do not try to generate any information.

                Context : {context}
                Question : {request.query}
                Answer: <|start_of_role|>assistant<|end_of_role|>"""

        response = watsonx_llm(prompt)

        return {
            "answer": response,
            "sources": [
                {
                    "relevance_score": doc["metadata"]["relevance_score"],
                    "metadata": doc["metadata"],
                }
                for doc in relevant_documents
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agentic-route")
async def agentic_route(query: QueryRequest):
    try:
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")


#         # models_available = {
#         #     "models_available": [
#         #         "codellama/codellama-34b-instruct-hf",
#         #         "google/flan-t5-xl",
#         #         "google/flan-t5-xxl",
#         #         "google/flan-ul2",
#         #         "ibm/granite-13b-instruct-v2",
#         #         "ibm/granite-20b-code-instruct",
#         #         "ibm/granite-20b-multilingual",
#         #         "ibm/granite-3-2-8b-instruct-preview-rc",
#         #         "ibm/granite-3-2b-instruct",
#         #         "ibm/granite-3-8b-instruct",
#         #         "ibm/granite-34b-code-instruct",
#         #         "ibm/granite-3b-code-instruct",
#         #         "ibm/granite-8b-code-instruct",
#         #         "ibm/granite-guardian-3-2b",
#         #         "ibm/granite-guardian-3-8b",
#         #         "meta-llama/llama-2-13b-chat",
#         #         "meta-llama/llama-3-1-70b-instruct",
#         #         "meta-llama/llama-3-1-8b-instruct",
#         #         "meta-llama/llama-3-2-11b-vision-instruct",
#         #         "meta-llama/llama-3-2-1b-instruct",
#         #         "meta-llama/llama-3-2-3b-instruct",
#         #         "meta-llama/llama-3-2-90b-vision-instruct",
#         #         "meta-llama/llama-3-3-70b-instruct",
#         #         "meta-llama/llama-3-405b-instruct",
#         #         "meta-llama/llama-guard-3-11b-vision",
#         #         "mistralai/mistral-large",
#         #         "mistralai/mixtral-8x7b-instruct-v01",
#         #     ]
#         # }

        llm = LLM(
            # model='watsonx/mistralai/mistral-large',
            model="watsonx/ibm/granite-3-8b-instruct",
            base_url=url,
            project_id=project_id,
            max_tokens=50,
            temperature=0.7,
            api_key=apikey,
        )

        collection_selector_agent = Agent(
            role="Collection Selector",
            goal="Analyze user queries and determine the most relevant ChromaDB collection.",
            backstory="Expert in query classification. Routes questions to the correct domain.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

        categorization_task = Task(
            description=f"""
            Based on the user query below, determine the best category.
            You must return ONLY one of these exact values: "Coffee", "Baseball", or "Dogs".
            If the query is unclear, choose the best match.

            User Query: "{query.query}"
            """,
            expected_output="Either 'Coffee', 'Baseball', or 'Dogs'.",
            agent=collection_selector_agent,
            # may need to use this to ensure correct response
            # output_pydantic=ClasificationResponse
        )

        crew = Crew(
            agents=[collection_selector_agent],
            tasks=[categorization_task],
            verbose=True,
        )

        crew_result = crew.kickoff()
        return {"category": crew_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_summary", response_model=GenerateSummaryResponse)
async def generate_summary(
    template_model: str = Body(default="generate_summary"),
    prompt_template_name: str = Body(default="generate_summary"),
    prompt_template_kwargs: Dict[str, str] = Body(...),
):
    """
    Endpoint to generate a summary based on provided data.
    """
    response = await generated_text_response(
        template_model, prompt_template_name, prompt_template_kwargs
    )
    result = {"generated_text": response}
    return result


@app.post("/generate_pet_name", response_model=GeneratePetNameResponse)
async def generate_pet_name(
    template_model: str = Body(default="pet_namer"),
    prompt_template_name: str = Body(default="pet_namer"),
    prompt_template_kwargs: Dict[str, str] = Body(...),
):

    response = await generate_json_response(
        template_model, prompt_template_name, prompt_template_kwargs
    )
    return response


async def generated_text_response(
    template_model: str,
    prompt_template_name: str,
    prompt_template_kwargs: Dict[str, Any],
) -> str:
    """
    Common functionality to generate a response using a specified model
    and prompt template.

    Args:
        template_model (str): The name of the models type to use.
        prompt_template_name (str): The name of the prompt template to use.
        prompt_template_kwargs (Dict[str, Any]): The keyword arguments for the prompt template.

    Returns:
        str: The generated text response from the model.
    """
    try:
        model_request = models.get(template_model)
        if not model_request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with name {template_model} not found",
            )

        prompt_template_request = prompt_templates.get(prompt_template_name)
        if not prompt_template_request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PromptTemplate with name {prompt_template_name} not found",
            )

        examples = examples_templates.get(
            prompt_template_name, ExamplesTemplate(template="")
        )
        if not examples:
            logging.warning(
                f"Examples for prompt template {prompt_template_name} not found. Using empty string."
            )

        model = model_request.get_model()
        prompt_template = PromptTemplate.from_template(
            template=prompt_template_request.template
        )
        output_parser = StrOutputParser()

        prompt_template_kwargs["examples"] = examples.template

        chain = prompt_template | model | output_parser
        generated_text = chain.invoke(prompt_template_kwargs)

        return generated_text
    except Exception as e:
        logging.error(f"Error in generate_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_json_response(
    template_model: str,
    prompt_template_name: str,
    prompt_template_kwargs: Dict[str, Any],
) -> dict:
    """
    Generate a JSON response using a specified model and prompt template.

    Args:
        template_model (str): The name of the language model to use.
        prompt_template_name (str): The name of the prompt template to use.
        prompt_template_kwargs (Dict[str, Any]): The keyword arguments for the prompt template.

    Returns:
        dict: A dictionary containing the generated response in JSON format.
              The response is nested under the key 'generated_text'.

    Raises:
        HTTPException: If the specified model or prompt template is not found,
                       or if there's an error during the generation process.
    """
    try:
        model_request = models.get(template_model)
        if not model_request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with name {template_model} not found",
            )

        prompt_template_request = prompt_templates.get(prompt_template_name)
        if not prompt_template_request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PromptTemplate with name {prompt_template_name} not found",
            )

        examples = examples_templates.get(
            prompt_template_name, ExamplesTemplate(template="")
        )
        if not examples:
            logging.warning(
                f"Examples for prompt template {prompt_template_name} not found. Using empty string."
            )

        model = model_request.get_model()
        prompt_template = PromptTemplate.from_template(
            template=prompt_template_request.template
        )
        output_parser = PydanticOutputParser(pydantic_object=JSONResponseTemplate)
        format_instructions = output_parser.get_format_instructions()

        prompt_template_kwargs["format_instructions"] = format_instructions
        prompt_template_kwargs["examples"] = examples.template

        chain = prompt_template | model | output_parser
        results = chain.invoke(prompt_template_kwargs)

        result = {"generated_text": results.dict()}
        return result
    except Exception as e:
        logging.error(f"Error in generate_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/agentic-route")
# async def agentic_route(query: QueryRequest):
#     try:
#         apikey = os.environ.get("IBM_APIKEY")
#         project_id = os.environ.get("PROJECT_ID")
#         url = os.environ.get("WATSON_URL")

#         # credentials = Credentials(
#         #     url=url,
#         #     api_key=apikey,
#         # )
#         # model_id = "ibm/granite-3-2b-instruct"
#         # parameters = {
#         #     "decoding_method": "sample",
#         #     "max_new_tokens": 500,
#         #     "temperature": 0.7,
#         #     "top_k": 50,
#         #     "top_p": 1,
#         #     "repetition_penalty": 1,
#         # }
#         # credentials = Credentials(url=url, api_key=apikey)
#         # ibm_model = Model(
#         #     model_id=model_id,
#         #     params=parameters,
#         #     credentials=credentials,
#         #     project_id=project_id,
#         # )

#         # class IBMWatsonLLM(LLM):
#         #     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         #         return ibm_model.generate_text(prompt=prompt, guardrails=False)

#         #     @property
#         #     def _llm_type(self) -> str:
#         #         return "ibm_watson"

#         #     def generate_text(self, prompt):
#         #         return self._call(prompt)

#         # llm = IBMWatsonLLM()
#         # models_available = {
#         #     "models_available": [
#         #         "codellama/codellama-34b-instruct-hf",
#         #         "google/flan-t5-xl",
#         #         "google/flan-t5-xxl",
#         #         "google/flan-ul2",
#         #         "ibm/granite-13b-instruct-v2",
#         #         "ibm/granite-20b-code-instruct",
#         #         "ibm/granite-20b-multilingual",
#         #         "ibm/granite-3-2-8b-instruct-preview-rc",
#         #         "ibm/granite-3-2b-instruct",
#         #         "ibm/granite-3-8b-instruct",
#         #         "ibm/granite-34b-code-instruct",
#         #         "ibm/granite-3b-code-instruct",
#         #         "ibm/granite-8b-code-instruct",
#         #         "ibm/granite-guardian-3-2b",
#         #         "ibm/granite-guardian-3-8b",
#         #         "meta-llama/llama-2-13b-chat",
#         #         "meta-llama/llama-3-1-70b-instruct",
#         #         "meta-llama/llama-3-1-8b-instruct",
#         #         "meta-llama/llama-3-2-11b-vision-instruct",
#         #         "meta-llama/llama-3-2-1b-instruct",
#         #         "meta-llama/llama-3-2-3b-instruct",
#         #         "meta-llama/llama-3-2-90b-vision-instruct",
#         #         "meta-llama/llama-3-3-70b-instruct",
#         #         "meta-llama/llama-3-405b-instruct",
#         #         "meta-llama/llama-guard-3-11b-vision",
#         #         "mistralai/mistral-large",
#         #         "mistralai/mixtral-8x7b-instruct-v01",
#         #     ]
#         # }

#         llm = LLM(
#             # model='watsonx/mistralai/mistral-large',
#             model='watsonx/ibm/granite-3-2b-instruct',
#             base_url=url,
#             project_id=project_id,
#             max_tokens=50,
#             temperature=0.7,
#             api_key=apikey
#         )


#         # Agent = Agent(
#         # role="{topic} specialist",
#         # goal="Figure {goal} out",
#         # backstory="I am the master of {role}",
#         # system_template="""<|start_header_id|>system<|end_header_id|>

#         # {{ .System }}<|eot_id|>""",
#         # prompt_template="""<|start_header_id|>user<|end_header_id|>

#         # {{ .Prompt }}<|eot_id|>""",
#         # response_template="""<|start_header_id|>assistant<|end_header_id|>

#         # {{ .Response }}<|eot_id|>""",
#         # )

#         collection_selector_agent = Agent(
#             role="Collection Selector",
#             goal="Analyze user queries and determine the most relevant ChromaDB collection.",
#             backstory="Expert in query classification. Routes questions to the correct domain.",
#             verbose=True,
#             allow_delegation=False,
#             llm=llm,
#         )

#         # categorization_task = Task(
#         #     description=f"""
#         #     Based on the user query below, determine the best category.
#         #     You must return ONLY one of these exact values: "Coffee", "Baseball", or "Jack_Russell_Terrier".
#         #     If the query is unclear, choose the best match.

#         #     User Query: "{query.query}"
#         #     """,
#         #     expected_output="Either 'Coffee', 'Baseball', or 'Jack_Russell_Terrier'.",
#         #     agent=collection_selector_agent,
#         # )

#         categorization_task = Task(
#             description=f"""
#             Based on the user query below, determine the best category.
#             You must return ONLY one of these exact values: "Coffee", "Baseball", or "Jack_Russell_Terrier".
#             If the query is unclear, choose the best match.

#             User Query: "{query.query}"
#             """,
#             expected_output="Either 'Coffee', 'Baseball', or 'Jack_Russell_Terrier'.",
#             agent=collection_selector_agent,
#         )

#         crew = Crew(
#             agents=[collection_selector_agent],
#             tasks=[categorization_task],
#             verbose=True,
#         )

#         crew_result = crew.kickoff()
#         return {"category": crew_result}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
