import argparse
import os
import logging
import glob
import dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain


def find_files(directory, wildcard):
    pattern = directory + '/' + wildcard
    files = glob.glob(pattern)
    return files


def get_pdf_text(pdf_docs):
    page_text = []
    for pdf in pdf_docs:
        for page in PdfReader(pdf).pages:
            page_text.append(page.extract_text())
    return ' '.join(page_text)


def get_text_chunks(text, chunk_size=1024, chunk_overlap=128):
    return CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    ).split_text(text)


def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to generate embeddings and prompt')

    # Add the --generate-embeddings argument with -g short form
    parser.add_argument('-g', '--generate-embeddings', action='store_true',
                        help='Generate embeddings (Run this first)')

    # Add the --prompt argument with -p short form
    parser.add_argument('-p', '--prompt', action='store_true', help='Ask questions')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    dotenv.load_dotenv()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parse_args()

    embeddings_model = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store_directory = 'vector_store'

    if args.generate_embeddings:
        logging.info('Generating embeddings...')

        # get pdf text
        pdf_docs = find_files('specs', '*.pdf')
        raw_text = get_pdf_text(pdf_docs)
        logging.info(f"training on {len(raw_text)} characters...")
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # for generating embeddings i use local model because it is free
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings_model)
        vectorstore.save_local(vector_store_directory)
        logging.info('vector store saved...')

    if args.prompt:
        vectorstore = FAISS.load_local(vector_store_directory, embeddings_model)
        logging.info('vector store loaded...')
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)

        while True:
            query = input("ask a question or type 'quit': ")
            logging.info(query)

            if query == 'quit':
                break

            if query == 'json':
                chat_memory = ConversationBufferWindowMemory(k=1)

                # this code is real hacky. there has to be a better way to transfer the memory
                # i tried just using the other memory object, but names are different between the two
                input_val = {}
                output_val = {}
                for message in memory.chat_memory.messages:
                    if message.type == 'human':
                        input_val['input'] = message.content

                    else:

                        output_val['output'] = message.content
                        logging.info(input_val)
                        logging.info(output_val)
                        chat_memory.save_context(input_val, output_val)

                llm = ChatOpenAI(temperature=0.0)

                conversation = ConversationChain(
                    llm=llm,
                    memory=chat_memory,
                    verbose=True,
                )
                output = conversation.predict(input='output all the messages and their required fields as a JSON string')
                print(output)
            else:
                result = qa({"question": query})
                print(result['answer'])


if __name__ == '__main__':
    main()
