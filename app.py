from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
#from langchain.chains.api import open_meteo_docs
#from langchain.chains import APIChain
from langchain.callbacks import get_openai_callback


import os

OPENAI_MODEL = "gpt-3.5-turbo-16k"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")



def map_reduce_func():
    map_prompt_template = """
                      Write a summary of this chunk of text that includes the main points and any important details.
                      {text}
                      """

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
                        Write a concise summary of the following text delimited by triple backquotes.
                        Return your response in bullet points which covers the key points of the text.
                        ```{text}```
                        BULLET POINT SUMMARY:
                        """

    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )















def number_of_videos(knowledge_base):
    standard_prompt = """"
        TASK: I NEED TO TRANSFORM THIS DOC IN A SERIES OF VIDEOS. HOW MANY VIDEO SCRIPTS DO YOU SUGGEST TO TRANSFORM THE CONTENT INTO VIDEO SERIES?
        Always follow strictly both OUTPUT EXAMPLE as your template for the answer and the output cannot be identical to the output examples as they are only guide and template for the desired result.

        ###OUTPUT EXAMPLE 1###
        Title: {title of the file}
        Language: The content is in PT_BR
        Thinking: This content is about a SOP of a food processing plant
        Thinking: This content describes the pasteurization of milk
        Suggestion: write 8 different scripts

        ###OUTPUT EXAMPLE 1###
        Title: {title of the file}
        Language: The content is in PT_BR
        Thinking: This content is about a SOP of a textile industry
        Thinking: This content revolves around a complex dyeing process
        Suggestion: write 5 different scripts

        """

    docs = knowledge_base.similarity_search(standard_prompt)
    # setup the chat model
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0)
    #llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=standard_prompt)
        print(cb)
    st.write(response)
    return (response)






def scripted_videos(knowledge_base, number_of_vids):

    standard_prompt = f""""
        Role: Act as a professional script writer
        {number_of_vids}
        Based on this information, scripts the content for bite-sized video content for Youtube.
        Each script must be short and must follow the example below:

        ###EXAMPLE###
                Vídeo 1: EPI

                **[CENAS - Máximo de 5 itens]**
                1. Apresentação dos equipamentos em uma mesa
                2. Close-up do funcionário colocando a sapatilha
                3. Close-up do funcionário vestindo touca descartável
                4. Close-up do funcionário vestido com luvas látex.

                **[NARRAÇÃO - Máximo de 5 itens]**
                1. Antes de entrar na sala de embalagem é necessário colocar os EPIs.
                2. Comece pela sapatilha para os pés,
                3. touca,
                4. e luvas látex para evitar contaminações.

                #SEPARADOR#
                Vídeo 2: ...

    """
    

    docs = knowledge_base.similarity_search("Etapas do procedimento ou processo para ser transformada em scripts de vídeo")
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
    #llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="Map-reduce")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=standard_prompt)
        print(cb)
    st.write(response)


    # Split the string at the separator
    videos = response.split('#SEPARADOR#')

    for video in videos:
        # Ensure there's content to display (in case there are extra separators)

        # Remove excessive spaces and newlines from the beginning and end
        cleaned_string = video.strip()

        # Split the first line from the rest of the string
        first_line, *remaining_lines = cleaned_string.split('\n')

        # Joining back the remaining lines into a single string
        video = '\n'.join(remaining_lines)


        if video.strip():
            video=video.strip()
            st.text_area(first_line, video)


def main():
    pdf = streamlit_interface()

    if pdf is not None:
        text_pdf = process_pdf(pdf)
        #knowledge_base = build_knowledge_base(text_pdf)
        #number_of_vids = number_of_videos(knowledge_base)
        #scripted_videos(knowledge_base, number_of_vids)






def streamlit_interface():
    load_dotenv()
    st.set_page_config(page_title="ChatGPT with PDF")
    st.header("ChatGPT with PDF 💬")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    return pdf


def process_pdf(pdf):
    # extract all the text from the PDF
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def build_knowledge_base(text):
    # split the text into chunks to build the knowledge graph
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base



if __name__ == '__main__':
    main()
