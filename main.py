import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import tiktoken
import getpass
import numpy as np

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("your_api_key")

st.title("KnoknLinker ")
file_path = 'faiss_store_googleai.pkl'

st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

# Replace OpenAI with ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro", max_tokens=500)  # You might need to set your Google API key

# Define embedding here
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['/n/n', '/n', '.'],  # Adjust punctuation separators as needed
        chunk_size=1000
    )

    main_placeholder.text("Data Loading...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Embeddings and saving data to FAISS index
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use Google Generative AI embeddings
    vectorstore_googleai = FAISS.from_documents(docs, embedding)  # Create FAISSIndex object
    print(docs)
    
    main_placeholder.text("Data Loading...Started...✅✅✅")
    time.sleep(2)
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_googleai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vectorstore = pickle.load(f)
            # Assuming `vectorstore` is your FAISS index
            # Perform similarity search with the query
            docs = vectorstore.similarity_search(query)
            print(docs[0].page_content , "sdfsfsffs")
            result  = llm.invoke(f" use the following portion of a long document to see if any of the text is relevent to answer the question. \n return any relevent text verbatim. \n {docs[0].page_content} \n  Question=```{query}```")
            # query_embedding = embedding.embed_query(query)  # Embed the query
            print(result.content)
            sources = result.content
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
            
    else:
        st.error("No existing knowledgebase found.")
# if query:
    # if os.path.exists(file_path):
    #     with open(file_path, 'rb') as f:
    #         vectorstore = pickle.load(f)
    #         # chain = RetrievalQAWithSourcesChain.from_llm(llm=llm.invoke(), retriever=vectorstore.as_retriever())  # Access index using .index
    #         # print(vectorstore.index)
    #         # st.write("Calling LLM...✅✅✅")
    #         # result = chain({'question': query}, return_only_outputs=True)

    #         # # Display the answer
    #         # st.subheader("Answer:")
    #         # st.write(result["answer"])
    #         # print(chain)
    #         result  = llm.invoke(query)
    #         print(result)
    #         # Display sources, if available 
    #         sources = result.content
    #         if sources:
    #             st.subheader("Sources:")
    #             sources_list = sources.split("\n")
    #             for source in sources_list:
    #                 st.write(source)
    # else:
    #     st.error("No existing knowledgebase found.")
 