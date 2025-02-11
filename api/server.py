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
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials


from langchain_ibm import WatsonxLLM

from crewai import Agent, Task, Crew, Process, LLM
from langchain.tools import tool


from pydantic import BaseModel, Field

import tempfile
import chromadb

from routes.models import ModelRequest

from schemas import (
    ExamplesTemplate,
    PromptTemplateRequest,
    JSONResponseTemplate,
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


# List of available models for watsonx
available_watsonx_models = {
    "models_available": [
        "codellama/codellama-34b-instruct-hf",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "google/flan-ul2",
        "ibm/granite-13b-instruct-v2",
        "ibm/granite-20b-code-instruct",
        "ibm/granite-20b-multilingual",
        "ibm/granite-3-2-8b-instruct-preview-rc",
        "ibm/granite-3-2b-instruct",
        "ibm/granite-3-8b-instruct",
        "ibm/granite-34b-code-instruct",
        "ibm/granite-3b-code-instruct",
        "ibm/granite-8b-code-instruct",
        "ibm/granite-guardian-3-2b",
        "ibm/granite-guardian-3-8b",
        "meta-llama/llama-2-13b-chat",
        "meta-llama/llama-3-1-70b-instruct",
        "meta-llama/llama-3-1-8b-instruct",
        "meta-llama/llama-3-2-11b-vision-instruct",
        "meta-llama/llama-3-2-1b-instruct",
        "meta-llama/llama-3-2-3b-instruct",
        "meta-llama/llama-3-2-90b-vision-instruct",
        "meta-llama/llama-3-3-70b-instruct",
        "meta-llama/llama-3-405b-instruct",
        "meta-llama/llama-guard-3-11b-vision",
        "mistralai/mistral-large",
        "mistralai/mixtral-8x7b-instruct-v01",
    ]
}


@app.get("/health")
async def health():
    """
    Health check endpoint to see if the API is up and running.
    """
    return {"message": "Fast API up!"}


chroma_client = chromadb.PersistentClient(path="./chroma_db")


@app.post("/process-documents-by-collection")
async def process_documents_by_collection(files: List[UploadFile] = File(...)):
    txt_files = []
    try:
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")
        if not (apikey and project_id and url):
            raise ValueError("Missing one or more required environment variables.")

        credentials = Credentials(
            url=url,
            api_key=apikey,
        )
        embedding_model = Embeddings(
            model_id="intfloat/multilingual-e5-large",
            credentials=credentials,
            project_id=project_id,
        )

        for upload in files:
            if upload.filename.endswith(".txt"):
                content = await upload.read()
                txt_files.append((upload.filename, content))
            else:
                print(f"Skipping file {upload.filename} as it is not a .txt file.")

        with tempfile.TemporaryDirectory() as temp_dir:
            for filename, content in txt_files:
                file_path = os.path.join(temp_dir, filename)
                file_name = os.path.splitext(filename)[0]

                with open(file_path, "wb") as buffer:
                    buffer.write(content)

                loader = TextLoader(file_path)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " "],
                    is_separator_regex=False,
                )
                documents = loader.load()
                splits = text_splitter.split_documents(documents)

                collection = chroma_client.get_or_create_collection(
                    name=file_name.lower(),
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 400,
                        "hnsw:M": 128,
                    },
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


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description=(
            "The user's input query that will be processed through a multi-agent pipeline "
            "to generate an intelligent answer."
        ),
    )


class CategoryResponse(BaseModel):
    category: str = Field(
        ...,
        description=(
            "The determined category for the query. It must be one of the following values: "
            "'technical', 'billing', or 'account', which help route the query to the correct domain."
        ),
    )


class FinalResponse(BaseModel):
    category: str = Field(
        ...,
        description=(
            "The category assigned to the query by the categorization agent. "
            "This value is one of 'technical', 'billing', or 'account'."
        ),
    )
    response: str = Field(
        ...,
        description=(
            "The final generated natural language answer. This response is produced after "
            "retrieving relevant documents and processing them via a dedicated generation agent."
        ),
    )


class AIQueryAnswerResponse(BaseModel):
    response: FinalResponse = Field(
        ...,
        description=(
            "The final response object containing both the query category and the generated answer, "
            "conforming to the multi-agent processing pipeline output."
        ),
    )


@app.post("/agentic-route")
async def agentic_route(query: QueryRequest):
    """
    Process a user query through a multi-agent pipeline to generate an intelligent answer.

    This endpoint orchestrates a three-step process:
      1. **Query Categorization**: An LLM-powered agent determines the query category from
         a fixed set ("technical", "billing", or "account").
      2. **Context Retrieval**: Based on the determined category, the system queries the
         corresponding ChromaDB collection using an embedding model to extract relevant documents.
      3. **Response Generation**: A dedicated agent generates a detailed answer using a structured
         prompt that incorporates the query and the retrieved document context.

    **Return Structure**:
      The response is a JSON object conforming to the `AIQueryAnswerResponse` model:
        {
            "response": {
                "category": <str>,  # one of "technical", "billing", or "account"
                "response": <str>   # the generated natural language answer
            }
        }

    **Raises**:
      - HTTPException: If an error occurs during processing.
    """
    try:
        apikey = os.environ.get("IBM_APIKEY")
        project_id = os.environ.get("PROJECT_ID")
        url = os.environ.get("WATSON_URL")

        categorization_llm = LLM(
            model="watsonx/ibm/granite-3-8b-instruct",
            base_url=url,
            project_id=project_id,
            max_tokens=50,
            temperature=0.7,
            api_key=apikey,
        )

        retrieval_llm = LLM(
            model="watsonx/ibm/granite-3-8b-instruct",
            base_url=url,
            project_id=project_id,
            max_tokens=1000,
            temperature=0.7,
            api_key=apikey,
        )

        generation_llm = LLM(
            model="watsonx/ibm/granite-3-8b-instruct",
            # model="watsonx/mistralai/mistral-large",
            base_url=url,
            project_id=project_id,
            max_tokens=3000,
            temperature=0.7,
            api_key=apikey,
        )

        collection_selector_agent = Agent(
            role="Collection Selector",
            goal="Analyze user queries and determine the most relevant ChromaDB collection.",
            backstory="Expert in query classification. Routes questions to the correct domain.",
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            llm=categorization_llm,
        )

        categorization_task = Task(
            description=f"""
            Based on the user query below, determine the best category.
            You must return ONLY one of these exact values: "technical", "billing", or "account".
            
            Category Definitions:
            - technical: Issues with system access, errors, API integration
            - billing: Questions about pricing, payments, invoices
            - account: User management, roles, organization settings
            
            IMPORTANT: Respond with EXACTLY ONE WORD from the list above.
            
            User Query: "{query.query}"
            """,
            expected_output="A JSON object with a 'category' field that must be either 'technical', 'billing', or 'account'",
            agent=collection_selector_agent,
            output_json=CategoryResponse,
            # may need to use this to ensure correct response
            # output_pydantic=CategoryResponse
        )

        @tool("query_collection_tool")
        def query_collection_tool(category: str, query: str) -> dict:
            """Tool to query ChromaDB based on category and return relevant documents"""

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

            query_embedding = embedding_model.embed_query(query)
            collection = chroma_client.get_collection(category.lower())
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )

            relevant_documents = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1 - distance
                if similarity > 0.8:  # should adjust? maybe?
                    metadata["collection"] = category.lower()
                    metadata["relevance_score"] = similarity
                    relevant_documents.append({"content": doc, "metadata": metadata})

            relevant_documents.sort(
                key=lambda x: x["metadata"]["relevance_score"], reverse=True
            )
            # lets see if 5 is enough
            relevant_documents = relevant_documents[:5]

            context = ""
            for doc in relevant_documents:
                score = doc["metadata"]["relevance_score"]
                content = doc["content"]
                context += f"\nRelevance Score: {score:.2f}\n{content}\n---\n"

            return {"category": category, "query": query, "context": context}

        retriever_agent = Agent(
            role="Category Retriever",
            goal="Query ChromaDB with the appropriate category and return results",
            backstory=(
                "You are responsible for taking the classified category and original query, "
                "querying the appropriate ChromaDB collection, and returning the results."
            ),
            verbose=True,
            allow_delegation=False,
            llm=retrieval_llm,
            max_iter=3,
            tools=[query_collection_tool],
        )

        retriever_task = Task(
            description=(
                "Take the category from the categorization task and the original query, "
                "use them to query the appropriate ChromaDB collection, and return the results. "
                f"Current query: {query.query}"
            ),
            expected_output=(
                "An object containing the category, query, and context from ChromaDB"
            ),
            agent=retriever_agent,
            context=[categorization_task],
        )

        # make a tool to check if the answer answers the question
        @tool("generate_response_tool")
        def generate_response_tool(context: str, query: str) -> dict:
            """Tool to generate a response using the specific prompt template"""
            prompt = f"""<|start_of_role|>system<|end_of_role|>
                        - You are a helpful, respectful, and honest assistant that can summarize long documents.
                        - Always respond as helpfully as possible, while being safe.
                        - Your responses should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
                        - Please ensure that your responses are socially unbiased and positive in nature.
                        - If a document does not make any sense, or is not factually coherent, explain why instead of responding something not correct.
                        - If you don't know the response to a query, please do not share false information.
                        <|start_of_role|>user<|end_of_role|>
                        You are an assistant for question-answering tasks. Generate a conversational response for the given question based on the given set of document context. Think step by step to answer in a crisp manner. Answer should not be more than 300 words. If you do not find any relevant answer in the given documents, please state you do not have an answer. Do not try to generate any information.
                        Context : {context}
                        Question : {query}
                        Answer: <|start_of_role|>assistant<|end_of_role|>"""

            return {"response": prompt}

        generation_agent = Agent(
            role="Response Generator",
            goal="Generate a comprehensive response using the specific prompt template",
            backstory=(
                "You are an expert at using structured prompts to generate precise, "
                "informative responses based on provided context."
            ),
            verbose=True,
            allow_delegation=False,
            llm=generation_llm,
            max_iter=3,
            tools=[generate_response_tool],
        )

        generation_task = Task(
            description=(
                "Using the context and query from the retriever task, generate a response using "
                "the specific prompt template via generate_response_tool. Return the complete response."
            ),
            # expected_output="A natural language response following the prompt template structure",
            expected_output="A JSON object with 'category' and 'response' fields, where 'response' contains the natural language answer",
            agent=generation_agent,
            context=[retriever_task],
            output_json=FinalResponse,
        )

        crew = Crew(
            agents=[collection_selector_agent, retriever_agent, generation_agent],
            tasks=[categorization_task, retriever_task, generation_task],
            process=Process.sequential,
            verbose=True,
        )

        crew_result = crew.kickoff()

        return {"response": crew_result}

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
