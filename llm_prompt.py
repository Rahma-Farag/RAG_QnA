from transformers import pipeline
import torch
from langchain.document_loaders import WikipediaLoader, OnlinePDFLoader, UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient

def get_prompt_embeddingllm():
  model_id = "HuggingFaceH4/zephyr-7b-beta"
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  prompt_in_chat_format = [
      {
          "role": "system",
          "content": """Using the information contained in the context,
  give a comprehensive answer to the question.
  Respond only to the question asked, response should be concise and relevant to the question.
  Provide the number of the source document when relevant.
  If the answer cannot be deduced from the context, do not give an answer.""",
      },
      {
          "role": "user",
          "content": """Context:
  {context}
  ---
  Now here is the question you need to answer.

  Question: {question}""",
      },
  ]
  RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
      prompt_in_chat_format, tokenize=False, add_generation_prompt=True
  )
  # embeddings
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  embedding_llm = SentenceTransformerEmbeddings(model_name=model_name)

  return RAG_PROMPT_TEMPLATE, embedding_llm

def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = [Document(page_content=x, metadata={"document": i}) for i, x in enumerate(text_splitter.split_text(text))]
    return docs

def get_answer(embedding_llm, RAG_PROMPT_TEMPLATE, docs, question):

  # vector database
  save_to_dir = "/content/wiki_chroma_db"
  vector_db = Chroma.from_documents(
      docs,
      embedding_llm,
      #persist_directory=save_to_dir
  )

  similar_docs = vector_db.similarity_search(question, k=1)


  retrieved_docs_text = [doc.page_content for doc in similar_docs]  # We only need the text of the documents
  context = "\nExtracted documents:\n"
  context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

  final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

  # Redact an answer
  # final_answer = READER_LLM(final_prompt)[0]["generated_text"]
  client = InferenceClient()
  response = client.text_generation(
      prompt=final_prompt,
      model="HuggingFaceH4/zephyr-7b-beta",
      temperature=0.8,
      max_new_tokens=500,
      seed=42,
      return_full_text=False,
  )
  
  return response

