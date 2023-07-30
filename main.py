import argparse
import logging
import dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
# from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in list(pdf_reader.pages):
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def parse_args():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to generate embeddings and prompt')

    # Add the --generate-embeddings argument with -g short form
    parser.add_argument('-g', '--generate-embeddings', action='store_true',
                        help='Generate embeddings')

    # Add the --prompt argument with -p short form
    parser.add_argument('-p', '--prompt', action='store_true', help='Prompt text')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    dotenv.load_dotenv()
    args = parse_args()

    if args.generate_embeddings:
        logging.info('Generating embeddings...')

        # get pdf text
        pdf_docs = ['BATS_FIX_Specification.pdf']
        raw_text = get_pdf_text(pdf_docs)
        logging.info(f"training on {len(raw_text)} characters...")
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create vector store
        # broker-spec-gpt = OpenAIEmbeddings()
        # broker-spec-gpt = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

        # for generating embeddings i use local model because it is free
        embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        # for questions i use openai
        embeddings = OpenAIEmbeddings

        # this code doesn't work yet.  i'd like to save out the above vector store so i don't
        # recreate it each time and then load it, but i'm getting an error.
        #      vectorstore = FAISS.load_local('broker.vector_store', embeddings)
        # vectorstore.load_local('broker.vector_store')
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # llm = ChatOpenAI(tempature=0)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)
        # qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever())
        # chat_history = []
        # query = "What is the version of the document?"
        # result = qa({"question": query, "chat_history": chat_history})
        while True:
            query = input("ask a question or type 'quit': ")
            logging.info(query)

            if query == 'quit':
                break

            logging.info('asking questions...')
            result = qa({"question": query})
            #
            # # create conversation chain
            # st.session_state.conversation = get_conversation_chain(
            #     vectorstore)
            logging.info(result['answer'])



if __name__ == '__main__':
    main()