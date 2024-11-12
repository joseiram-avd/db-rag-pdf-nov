# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## 0-Compute Sample Configurations
# MAGIC
# MAGIC 1. **Access Mode**: Assigned to single user
# MAGIC 2. **Databricks Runtime Version**: 15.4 LTS
# MAGIC 3. Photon not required
# MAGIC 4. **Single-Node Cluster with No Worker**
# MAGIC 5. **Driver/Worker Type**: m5d.4xlarge (64GB Memory, 16 Cores)
# MAGIC 6. **Terminate** after 60 minutes of inactivity

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1-Set up
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Please remember to modify config notebook before proceeding

# COMMAND ----------

# DBTITLE 1,Install required external libraries 
# MAGIC %pip install --quiet -U transformers==4.41.1 pypdf==4.1.0 langchain-text-splitters==0.2.0 databricks-vectorsearch mlflow tiktoken==0.7.0 torch==2.3.0 llama-index==0.10.43
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %run ./00-init-advanced $reset_all_data=true

# COMMAND ----------

spark.sql(f"USE {catalog}.{dbName}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{dbName}.{volumeName}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2-PDF Ingestion
# MAGIC

# COMMAND ----------

volume_folder =  f"/Volumes/{catalog}/{dbName}/{volumeName}"

# COMMAND ----------

dbutils.fs.mkdirs(f"{volume_folder}/{folderName}")

# COMMAND ----------

folderVolumePath = f"{volume_folder}/{folderName}"
print(folderVolumePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Please Upload Some PDFs to the folder in the UC volume above
# MAGIC

# COMMAND ----------

# Run this cell if there's no sample PDFs for testing. This will upload a number of Databricks Docs PDF
upload_pdfs_to_volume(folderVolumePath)

# COMMAND ----------

display(dbutils.fs.ls(folderVolumePath))

# COMMAND ----------

# DBTITLE 1,Ingesting PDF files as binary format using Auto Loader
df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobFilter", "*.pdf")
        .load('dbfs:'+folderVolumePath))

# Write the data as a Delta table
(df.writeStream
  .trigger(availableNow=True)
  .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/raw_docs')
  .table('pdf_raw').awaitTermination())

# COMMAND ----------

# MAGIC %sql SELECT * FROM pdf_raw

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3-PDF Content Extraction
# MAGIC

# COMMAND ----------

# DBTITLE 1,To extract our PDF,  we'll need to setup libraries in our nodes
import warnings
from pypdf import PdfReader

def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)
        parsed_content = [page_content.extract_text() for page_content in reader.pages]
        return "\n".join(parsed_content)
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None

# COMMAND ----------

# DBTITLE 1,Trying our text extraction function with a single pdf file
import io
import re
with requests.get('https://github.com/databricks-demos/dbdemos-dataset/blob/main/llm/databricks-pdf-documentation/Databricks-Customer-360-ebook-Final.pdf?raw=true') as pdf:
  doc = parse_bytes_pypdf(pdf.content)  
  print(doc)

# COMMAND ----------

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer to match our model size (will stay below gte 1024 limit)
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=10)
    def extract_and_split(b):
      txt = parse_bytes_pypdf(b)
      if txt is None:
        return []
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# DBTITLE 1,Using Databricks Foundation model GTE as embedding endpoint
from mlflow.deployments import get_deploy_client

# gte-large-en Foundation models are available using the /serving-endpoints/databricks-gtegte-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint=embeddings_endpoint, inputs={"input": ["What is Apache Spark?"]})
pprint(embeddings)

# COMMAND ----------

# DBTITLE 1,Create the final databricks_pdf_documentation table containing chunks and embeddings
# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS pdf_content_embeddings (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4-Generate Embeddings for PDFs

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint=embeddings_endpoint, inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

(spark.readStream.table('pdf_raw')
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_embedding("content"))
      .selectExpr('path as url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/pdf_chunk')
    .table('pdf_content_embeddings').awaitTermination())

#Let's also add our documentation web page content
if table_exists(f'{catalog}.{dbName}.databricks_documentation'):
  (spark.readStream.option("skipChangeCommits", "true").table('databricks_documentation') #skip changes for more stable demo
      .withColumn('embedding', get_embedding("content"))
      .select('url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/docs_chunks')
    .table('pdf_content_embeddings').awaitTermination())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM pdf_content_embeddings limit 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from pdf_content_embeddings;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5-Create & Sync Vector Search Index

# COMMAND ----------

# DBTITLE 1,Creating the Vector Search endpoint
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# DBTITLE 1,Create the Self-managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{dbName}.pdf_content_embeddings"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{dbName}.{vectorSearchIndexName}"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

# COMMAND ----------

# MAGIC %md
# MAGIC # How Hybrid Search Works

# COMMAND ----------

# DBTITLE 1,ann search (default)
question = "What is the best way to connect to BigQuery?"

response = deploy_client.predict(endpoint=embeddings_endpoint, inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["id", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
pprint(docs)

# COMMAND ----------

# DBTITLE 1,hybrid search
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text= question,
  query_vector=embeddings[0],
  columns=["id", "content"],
  num_results=1,
  query_type="hybrid")
docs = results.get('result', {}).get('data_array', [])
pprint(docs)
