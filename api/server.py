import json
import os
from typing import Dict, List, Any
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
from ibm_watsonx_ai import Credentials

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
                chunk_size=300,
                chunk_overlap=100,
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
    Simple search.
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

        results = collection.query(query_embeddings=query_embedding, n_results=1)

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
