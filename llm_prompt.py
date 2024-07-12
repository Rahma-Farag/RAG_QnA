from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.document_loaders import WikipediaLoader, OnlinePDFLoader, UnstructuredURLLoader

from langchain.text_splitter import NLTKTextSplitter
# import nltk
# nltk.download('punkt')

from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_community.vectorstores import Chroma

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs

def get_answer(READER_LLM, RAG_PROMPT_TEMPLATE, docs, question):

  doc_content = [ doc.page_content for doc in docs]
  metadatas = [ {"document":i} for i in range(len(docs))]
  # # splitter
  # text_splitter = NLTKTextSplitter(chunk_size=60, chunk_overlap=5)
  # tokens_chunks = text_splitter.create_documents(
  #     doc_content,
  #     metadatas=metadatas
  # )

  # embeddings
  model_name = "sentence-transformers/all-MiniLM-L6-v2"

  embedding_llm = SentenceTransformerEmbeddings(model_name=model_name)

  # vector database
  save_to_dir = "/content/wiki_chroma_db"
  vector_db = Chroma.from_documents(
      docs,
      embedding_llm,
      persist_directory=save_to_dir
  )

  similar_docs = vector_db.similarity_search(question, k=1)


  retrieved_docs_text = [doc.page_content for doc in similar_docs]  # We only need the text of the documents
  context = "\nExtracted documents:\n"
  context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

  final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

  # Redact an answer
  final_answer = READER_LLM(final_prompt)[0]["generated_text"]

  # final_answer = map_reduce_chain({"input_documents": similar_docs,
  #                                 "question": question
  #                                 }, return_only_outputs=True)
  return final_answer


def get_llm(READER_MODEL_NAME = "stabilityai/stablelm-2-zephyr-1_6b"):

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)#, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

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

    return READER_LLM, RAG_PROMPT_TEMPLATE
