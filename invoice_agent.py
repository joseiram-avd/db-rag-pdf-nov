
import mlflow
import mlflow.pyfunc
import pandas as pd
from typing import Dict, List, Any, Optional, Generator
import json
import os
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk.service.serving import DataframeSplitInput

from mlflow.pyfunc import ChatAgent

from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatAgentChunk,
    ChatContext
)

from databricks import agents
from mlflow.models import ModelConfig
from mlflow.models.resources import DatabricksServingEndpoint

import uuid  # Add at top of file
from uuid import UUID

class InvoiceProcessingAgent(ChatAgent):
# class InvoiceProcessingAgent(mlflow.pyfunc.PythonModel):
    
    def __init__(self, endpoint_name: str = "databricks-claude-3-7-sonnet"):
        """
        Initialize the Invoice Processing Agent.
        
        Args:
            endpoint_name: Name of the Databricks Foundation Model endpoint to use
        """
        self.endpoint_name = endpoint_name
        self.workspace_client = WorkspaceClient()
        self.prompts = self._load_static_prompts()
        
        # Initialize OpenAI client for the foundation model
        self.openai_client = self.workspace_client.serving_endpoints.get_open_ai_client()
        
        # Configure agent parameters
        self.config = ModelConfig(development_config={
            "endpoint_name": endpoint_name,
            "temperature": 0.0,
            "max_tokens": 4096
        })
    
    def _load_static_prompts(self) -> Dict[str, str]:
        """Load the prompts from JSON"""
        json_format = {
               "Empresa": "{{Do topo da página extrair o Nome da Empresa}}",
               "CNPJ": "{{Do topo da página extrair o CNPJ}}",
               "Endereço": "{{Do topo da página extrair o Endereço Completo da Empresa}}",
               "Data Emissao": "{{Do topo da Pagina campo Date}}",
               "Invoice Nº": "{{Numero seguido de Ano}}",
               "Bill_to": "{{Campo Bill To}}",
               "Bill_Address": "{{Linhas entre Bill To e P.IVA}}",
               "P.IVA": "{{Código IVA}}"
               }
        
        prompts = {
            "document_classification": "Classify this document into one of the following categories: invoice, receipt, ID, contract.",
            "information_extraction": f"Extract the following fields from this document: Empresa, CNPJ, Endereço, Data Emissao, Invoice Nº, Bill_to, Bill_Address, P.IVA. If the field is not present, return 'N/A'.Your response must be a valid JSON (no extra text) object starting with {{ and ending with }}. The JSON should be formatted as follows (no extra text): {json_format}."
            }
                
        return prompts
    
    def predict(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None
    ) -> ChatAgentResponse:
        """
        Process a user query and return a response according to ChatAgent interface.
        
        Args:
            messages: List of ChatAgentMessage objects
            context: Optional ChatContext object
            custom_inputs: Optional additional inputs
            
        Returns:
            ChatAgentResponse containing the agent's response
        """
        # Add unique ID
        message_id = str(uuid.uuid4())

        # Extract document text from the user message
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            return ChatAgentResponse(messages=[
                ChatAgentMessage(id=message_id, role="assistant", content="Please provide a document to process.")
            ])
        
        document_text = user_messages[-1].content
        
        # Process the document
        # document_class = self._classify_document(document_text)
        extracted_info = self._extract_information(document_text)
        
        # Format the response
        # result = {
        #     "document_class": document_class,
        #     "extracted_information": extracted_info
        # }
        
        # Create and return the agent response
        return ChatAgentResponse(messages=[
            ChatAgentMessage(
                id=message_id,
                role="assistant", 
                # content=f"Document classified as: {document_class}\n\nExtracted information: {json.dumps(extracted_info, indent=2)}"
                content=f"{json.dumps(extracted_info, indent=2)}"
            )
        ])
    
    def predict_stream(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None
    ) -> Generator[ChatAgentChunk, None, None]:
        """
        Stream the response for a user query according to ChatAgent interface.
        
        Args:
            messages: List of ChatAgentMessage objects
            context: Optional ChatContext object
            custom_inputs: Optional additional inputs
            
        Yields:
            ChatAgentChunk objects containing parts of the response
        """
        message_id = str(uuid.uuid4())

        # Extract document text from the user message
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            yield ChatAgentChunk(delta=ChatAgentMessage(
                id=message_id, 
                role="assistant", 
                content="Please provide a document to process."
            ))
            return
        
        document_text = user_messages[-1].content
        
        # Process the document (in real implementation, this would be streamed)
        # document_class = self._classify_document(document_text)
        
        # Yield the classification result
        # yield ChatAgentChunk(delta=ChatAgentMessage(
        #     id=message_id, 
        #     role="assistant", 
        #     content=f"Document classified as: {document_class}\n\n"
        # ))
        
        # Process and yield the extraction result
        extracted_info = self._extract_information(document_text)
        yield ChatAgentChunk(delta=ChatAgentMessage(
            id=message_id, 
            role="assistant", 
            # content=f"Extracted information: {json.dumps(extracted_info, indent=2)}"
            content=f"{json.dumps(extracted_info, indent=2)}"
        ))
    
    # def _classify_document(self, document_text: str) -> str:
    #     """
    #     Classify the document using foundation model.
    #     """
    #     classification_prompt = self.prompts["document_classification"]
        
    #     response = self.openai_client.chat.completions.create(
    #         model=self.endpoint_name,
    #         messages=[
    #             {"role": "system", "content": "You are an expert document classifier."},
    #             {"role": "user", "content": f"{classification_prompt}\n\nDocument: {document_text}"}
    #         ],
    #         temperature=0.0,
    #         max_tokens=256
    #     )
        
    #     return response.choices[0].message.content.strip()
    
    def _extract_information(self, document_text: str) -> Dict[str, Any]:
        """
        Extract information from the document using foundation model.
        """
        extraction_prompt = self.prompts["information_extraction"]
        
        response = self.openai_client.chat.completions.create(
            model=self.endpoint_name,
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured information from documents."},
                {"role": "user", "content": f"{extraction_prompt}\n\nDocument: {document_text}"}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        
        try:
            # Try to parse as JSON
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # If not valid JSON, return as text
            return {"raw_extraction": response.choices[0].message.content.strip()}


AGENT = InvoiceProcessingAgent()

mlflow.models.set_model(AGENT)
