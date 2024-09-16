# Databricks notebook source
# MAGIC %md
# MAGIC ## 1-Set up

# COMMAND ----------

# MAGIC %pip install databricks-agents 'mlflow>=2.13'
# MAGIC %pip install lxml==4.9.3 langchain==0.1.5 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.18.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00-helper

# COMMAND ----------

persona_prompt = "You are a Big Data chatbot. Please answer Big Data question only. If you don't know or not related to Big Data, don't answer."

role_prompt = "You are a trustful assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, AI, ML, Datawarehouse, platform, API or infrastructure, Cloud administration question related to Databricks. "

guardrail_prompt = "You are classifying documents to know if this question is related with Databricks in AWS, Azure and GCP, Workspaces, Databricks account and cloud infrastructure setup, Data Science, Data Engineering, Big Data, Datawarehousing, SQL, Python and Scala or something from a very different field. If the prompt contains the word Databricks, answer Yes. Answer no if it's not related with the topics mentioned above. Also answer no if the last part is inappropriate."

guardrail_example = """
Question: What is Databricks?
Expected Response: Yes
"""

# COMMAND ----------

spark.sql(f"USE {catalog}.{dbName}")

# COMMAND ----------

import re
import pandas as pd
import mlflow
import json

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2-Langchain Test

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
# MAGIC ## 3-Add Conversation History

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
# MAGIC ## 4-Add Filter/Guardrail

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

# MAGIC %md
# MAGIC ## 5-Retrieve Content from Vector Search DB

# COMMAND ----------

spark.sql(f'GRANT USAGE ON CATALOG `{catalog}` TO `{yourEmailAddress}`');
spark.sql(f'GRANT USAGE ON DATABASE `{catalog}`.`{dbName}` TO `{yourEmailAddress}`');

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
WorkspaceClient().grants.update(c.SecurableType.TABLE, f"{catalog}.{dbName}.{vectorSearchIndexName}",
changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal=f"{yourEmailAddress}")])

# COMMAND ----------

index_name=f"{catalog}.{dbName}.{vectorSearchIndexName}"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
# from langchain.chains import RetrievalQA
import os

# os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(secret_scope_name, secret_key_name)

embedding_model = DatabricksEmbeddings(endpoint=embeddings_endpoint)

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    # vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vsc = VectorSearchClient(workspace_url=host)
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
# MAGIC ## 6-Incorporate Vector Search Content in Prompt

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

# MAGIC %md
# MAGIC ## 7-Finished Chain and Query Testing
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
    return "!@#$".join([d.page_content for d in docs])

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
    "context": itemgetter("context"),
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "context": itemgetter("context"),
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
        {"role": "user", "content": followup_question + " Give me an extended and detailed response"}
    ]
}
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 8-Register the chatbot model to Unity Catalog

# COMMAND ----------

import cloudpickle
import langchain
import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{dbName}.{finalchatBotModelName}"

mlflow.langchain.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_inputs_outputs=True,
    registered_model_name=model_name
)

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

# MAGIC %md
# MAGIC ## 9-Inference

# COMMAND ----------

# MAGIC %md Let's try loading our model

# COMMAND ----------

print(model_info.model_uri)

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10-Evaluation

# COMMAND ----------

# Get the run ID of the logged model
run_id = model.run_id

# Download the inference inputs and outputs log
client = mlflow.tracking.MlflowClient()
artifact_path = f"runs:/{run_id}/inference_inputs_outputs.json"
local_path = client.download_artifacts(run_id, "inference_inputs_outputs.json")

# Read and print the log
with open(local_path, 'r') as f:
    logs = json.load(f)
    print(json.dumps(logs, indent=2))

# COMMAND ----------

evaluation_data = []

for item in logs['data']:
  unit_request = {}
  unit_request["request_id"] = item[logs['columns'].index("session_id")][0]
  unit_request["request"] = item[logs['columns'].index("input-messages")][0]['content']
  unit_request["response"] = item[logs['columns'].index("output-result")]
  unit_request["retrieved_context"] = []
  try:
    context_parts = item[logs['columns'].index("output-context")].split('!@#$')
  except AttributeError:
    continue
  for index, unit_uri in enumerate(item[logs['columns'].index("output-sources")]):
    unit_request['retrieved_context'].append({"content": context_parts[index], "doc_uri": unit_uri})
  evaluation_data.append(unit_request)

# COMMAND ----------

evaluation_data_df = pd.DataFrame(evaluation_data)

# COMMAND ----------

display(spark.createDataFrame(evaluation_data))

# COMMAND ----------

# If you do not start a MLflow run, `evaluate(...) will start a Run on your behalf.
with mlflow.start_run(run_name="rag_poc_eval_run"):
  evaluation_results = mlflow.evaluate(data=evaluation_data_df, model_type="databricks-agent")

# COMMAND ----------

metrics_as_dict = evaluation_results.metrics

print("Aggregate metrics computed:")
display(metrics_as_dict)


# COMMAND ----------

per_question_results_df = evaluation_results.tables['eval_results']
display(per_question_results_df)

print("Columns available for each question:")
print(per_question_results_df.columns)

