# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Configure PAT Token for Development Purposes
# MAGIC
# MAGIC 1. Open up Web Terminal on your cluster
# MAGIC 2. Command: `databricks configure`
# MAGIC 	1. Host: https://YOUR_WORKSPACE_URL (Ending in databricks.com)
# MAGIC 	2. username: Your Email Address
# MAGIC 	3. Password: Your Personal Access Token (Create one by following the instructions [here](https://docs.databricks.com/en/dev-tools/auth/pat.html#databricks-personal-access-tokens-for-workspace-users))
# MAGIC
# MAGIC     **Note: For Production Purposes, We recommend service pinricpal token instead of PAT token**
# MAGIC
# MAGIC 2. Command: `databricks secrets create-scope --scope [secret_scope_name]`
# MAGIC 3. Command: `databricks secrets list --scope [secret_scope_name]`
# MAGIC 4. Command: `databricks secrets put --scope [secret_scope_name] --key [secrete_key_name]`
# MAGIC 	1. `select-editor` (Select vim)
# MAGIC 	2. Paste the PAT token you created above
# MAGIC 	3. :wq (Save the info and exit from vim)

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC %pip install pip mlflow[databricks]==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00-helper

# COMMAND ----------

chatBotModel = "databricks-dbrx-instruct"
max_tokens = 2000
VECTOR_SEARCH_ENDPOINT_NAME = "one-env-shared-endpoint-8"
vectorSearchIndexName = "pdf_content_embeddings_index"
embeddings_endpoint = "databricks-bge-large-en"
catalog = "hz_demos_rag_pdf_databricks"
dbName = "hz_demos_rag_pdf_db"
secret_scope_name = "hzdemos"
secret_key_name = "hz_rag_sp_token"
finalchatBotModelName = "hz_rag_pdf_bot"

# COMMAND ----------

persona_prompt = "You are a Big Data chatbot. Please answer Big Data question only. If you don't know or not related to Big Data, don't answer."

role_prompt = "You are a trustful assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, AI, ML, Datawarehouse, platform, API or infrastructure, Cloud administration question related to Databricks. "

guardrail_prompt = "You are classifying documents to know if this question is related with Databricks in AWS, Azure and GCP, Workspaces, Databricks account and cloud infrastructure setup, Data Science, Data Engineering, Big Data, Datawarehousing, SQL, Python and Scala or something from a very different field. Also answer no if the last part is inappropriate."

guardrail_example = """
Question: What is Databricks?
Expected Response: Yes
"""

# COMMAND ----------

spark.sql(f"USE {catalog}.{dbName}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploring Langchain capabilities

# COMMAND ----------

# DBTITLE 1,Spark Chat Model Prompt
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint=chatBotModel, max_tokens = max_tokens)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What is Spark?"}))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Adding conversation history to the prompt 

# COMMAND ----------

prompt_with_history_str = f"{persona_prompt}" + """
Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

# DBTITLE 1,Chat History Extractor Chain
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
}))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Add a filter on top to only answer Databricks-related questions.

# COMMAND ----------

# DBTITLE 1,Checkpoint: Databricks Inquiry Classifier
chat_model = ChatDatabricks(endpoint=chatBotModel, max_tokens = max_tokens)

guardrail_str = f"{guardrail_prompt}. Here are some examples: {guardrail_example}"

is_question_about_databricks_str = guardrail_str + """
Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_databricks_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_databricks_str
)

is_about_databricks_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about Databricks: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about Databricks
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is today's weather"}
    ]
}))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Use LangChain to retrieve documents from the vector store

# COMMAND ----------

spark.sql('GRANT USAGE ON CATALOG `hz_demos_rag` TO `h.zhang@databricks.com`');
spark.sql('GRANT USAGE ON DATABASE `hz_demos_rag`.`hz_rag_ecl` TO `h.zhang@databricks.com`');

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
WorkspaceClient().grants.update(c.SecurableType.TABLE, f"{catalog}.{dbName}.{vectorSearchIndexName}",
changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="h.zhang@databricks.com")])

# COMMAND ----------

index_name=f"{catalog}.{dbName}.{vectorSearchIndexName}"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

#Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
test_demo_permissions(host, secret_scope=secret_scope_name, secret_key=secret_key_name, vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name=embeddings_endpoint, managed_embeddings = False)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
import os

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(secret_scope_name, secret_key_name)

embedding_model = DatabricksEmbeddings(endpoint=embeddings_endpoint)

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model, columns=["url"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What is Apache Spark?"}]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improve document search using LLM to generate a better sentence for the vector store, based on the chat history

# COMMAND ----------

# DBTITLE 1,Contextual Query Generation Chain
from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Final Query
# MAGIC
# MAGIC

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = role_prompt + """
If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about Databricks.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_databricks_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

# DBTITLE 1,Final Query Engine
initial_question = input("Question?: ")
dialog = {
    "messages": [
        {"role": "user", "content": initial_question + " Give me an extended and detailed response"}
        
        # {"role": "assistant", "content": initial_answer}, 
        # {"role": "user", "content": final_question}
    ]
}
# print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

followup_question = input("Question?: ")
dialog = {
    "messages": [
        {"role": "user", "content": dialog['messages'][0]['content']},
        {"role": "assistant", "content": response['result']}, 
        {"role": "user", "content": followup_question}
    ]
}
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Register the chatbot model to Unity Catalog

# COMMAND ----------

import cloudpickle
import langchain
import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{dbName}.{finalchatBotModelName}"

with mlflow.start_run(run_name=f"{finalchatBotModelName}_run") as run:
    #Get our model signature from input/output
    output = full_chain.invoke(dialog)
    signature = infer_signature(dialog, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=dialog,
        signature=signature,
        example_no_conversion=True,
    )

# COMMAND ----------

# MAGIC %md Let's try loading our model

# COMMAND ----------

initial_question = input("Question?: ")
dialog = {
    "messages": [
        {"role": "user", "content": initial_question + " Give me an extended and detailed response"}
    ]
}
print(dialog)

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model_response = model.invoke(dialog)

# COMMAND ----------

display_chat(dialog["messages"], model_response)
