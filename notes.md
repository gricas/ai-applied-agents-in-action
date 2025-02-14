# AI Applied: Agents in Action Tutorial

---

## Introduction

**Welcome to AI Applied: Agents in Action!**

Hi everyone, I'm David Levy, a Solution Architect from IBM. In this tutorial, I'll show you how to seamlessly integrate multiple AI agents into your application. We'll walk through a practical example that covers query categorization, context retrieval from a vector DB, and natural language response generation—all using a multi-agent approach.

This session is designed to give you a clear, step-by-step guide on working with agents in your projects. Let's dive in and explore how you can leverage these tools to build smarter applications.

Let's get started!

---

## Repository Setup

Go to the GitHub repository in the description of the video and clone the repo to your computer.

- **Task:** `Clone the repository from GitHub`

---

## UI Setup

Navigate to the UI directory and install the dependencies:

- **Task:** `cd ui`
  - **Subtask:** `npm i` *(Install root dependencies)*
  - **Subtask:** `npm run setup` *(This will take a minute or two)*


This application uses React and Express (both written in TypeScript) and utilizes the Carbon Design Framework for styling. We won't be modifying the UI in this tutorial, but the code is modular, so feel free to extend it as needed.

<***We cut here while waiting for the deps to install*** >

- **Task:** Copy environment variables for both client and server:
  - **Subtask:** Copy `.env.example` to `.env`

---

## API Setup

After the UI dependencies are installed, return to the root of the codebase and change directories into the API folder:

- **Task:** `cd api`
- **Task:** Create a virtual environment
- **Task:** `pyenv virtualenv aiagentic`
- **Task:** `pyenv activate aiagentic`
- **Task:** `pip install --use-deprecated=legacy-resolver -r requirements.txt` *(This will take a minute or two)*

<***We cut here while waiting for the deps to install*** >

- **Task:** `cp .env.example .env` 

The API is built using FastAPI (in Python), which comes with built-in Swagger support.

<***May have been logged out here when grabbing API key from cloud.ibm.com*** >
- **Task:** Get API Key, Project, and URL from watsonx.ai

---

## Application Initialization

Checkout branch `01_Step` and initialize the services to see the initial browser view:

- **Task:** `git checkout 01_Step`
- **Task:** Lets **start up**  all our services!



- **Task:** `python scripts/process_documents.py` *(Populate the ChromaDB)*
- **Task:** Walk through `process_documents.py` to explain how collections are created based on file names.

---

## Categorization Agent
<***First Section, longer than the following two sections, due to little copy and paste*** >

So now lets build our first agent. If we look at the doc string in the route, the first thing we must do is categorize the query and return a value that is either billing, technical, or account.

We will be using the **CrewAI** agent framework. Import the following classes from the `crewai` package:

- `Agent`, `Task`, `Crew`, `Process`, `LLM`

- **Task:** Build categorization agent

```python

class CategoryResponse(BaseModel):
    category: str = Field(
        ...,
        description=(
            "The determined category for the query. It must be one of the following values: "
            "'technical', 'billing', or 'account', which help route the query to the correct domain."
        )
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

        categorization_agent = Agent(
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
            agent=categorization_agent,
            output_json=CategoryResponse,
        )

        crew = Crew(
            agents=[categorization_agent],
            tasks=[categorization_task],
            process=Process.sequential,
            verbose=True
        )

        category_result = crew.kickoff()

        print(category_result)
        crew_result = {
            "json_dict": {
                "response": "This WILL be generated by our multi agent RAG process",
                "category": category_result['category'],
            }
        }

        return {"response": crew_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---


<***End of first section*** >

## Retriever Agent

Now that we have the basic categorization agent in place, we move on to enhancing the pipeline. In branch 02_Step, we integrate the retrieval functionality. This part of the tutorial shows how to use a tool within the agent to query ChromaDB for relevant documents based on the category determined earlier.

- **Task:** `git checkout 02_Step`

```python

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

        retriever_llm = LLM(
            model="watsonx/ibm/granite-3-8b-instruct",
            base_url=url,
            project_id=project_id,
            max_tokens=1000,
            temperature=0.7,
            api_key=apikey,
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
            llm=retriever_llm,
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


        crew = Crew(
            agents=[categorization_agent, retriever_agent],
            tasks=[categorization_task, retriever_task],
            process=Process.sequential,
            verbose=True
        )

        category_result = crew.kickoff()

        print(category_result)
        crew_result = {
            "json_dict": {
                "response": "This WILL be generated by our multi agent RAG process",
                "category": "Bye for now" 
            }
        }

        return {"response": crew_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

```

---


<***End of second section*** >

## Generation Agent

In the final step, we implement the generation agent. This agent is responsible for synthesizing the final natural language answer by combining the query and the context retrieved from the previous steps. We'll use a specialized prompt template to generate a comprehensive response.

<***I had to go to cloud.ibm.com or watsonx.ai where I was logged out during this section, when showing the prompt template I took from Watson Studio***>

- **Task:** `git checkout 03_Step`

```python

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

        generation_llm = LLM(
            model="watsonx/ibm/granite-3-8b-instruct",
            # model="watsonx/mistralai/mistral-large",
            base_url=url,
            project_id=project_id,
            max_tokens=3000,
            temperature=0.7,
            api_key=apikey,
        )

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
            expected_output="A JSON object with 'category' and 'response' fields, where 'response' contains the natural language answer",
            agent=generation_agent,
            context=[retriever_task],
            output_json=FinalResponse,
        )

        crew = Crew(
            agents=[categorization_agent,
                    retriever_agent, generation_agent],
            tasks=[categorization_task, retriever_task, generation_task],
            process=Process.sequential,
            verbose=True,
        )

        crew_result = crew.kickoff()

        return {"response": crew_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

```

---

<***End of tutorial***>


## Sign Off

Awesome! We've built a pretty sophisticated multi-agent pipeline here. So just to recap, we built the backend to an agentic RAG chatbot, that is able to identify and the query's category, target the correct ChromaDB collection, and interpolate the query and the context into a custom prompt and generate a natural language response.

So with this application and process, we would love for you to explore additional use cases, customize the UI, and experiment with the CrewAI framework and build some cool stuff. Maybe add a route that makes a web search if the query is out of bounds of the db? Maybe format an Agent who's job it is is only to format the response in a particular way? We would love to see anything you do with it.

Dive into the code, have fun, and build something cool, refactor make it better. Just be creative!

Thanks for spending the time with me today, hope to see you soon!
