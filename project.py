from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import streamlit as st
from langchain.schema import StrOutputParser
import os

OPENAI_MODEL = "gpt-4-1106-preview"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def main():
    pdf = streamlit_interface()

    if pdf is not None:
        text_pdf = process_pdf(pdf)
        steps = map_reduce_func(text_pdf)
        st.write(steps)
        response_content = scripted_videos(steps)

        # Split the string at the separator
        videos = response_content.split("#SEPARADOR#")

        for video in videos:
            # Ensure there's content to display (in case there are extra separators)

            # Remove excessive spaces and newlines from the beginning and end
            cleaned_string = video.strip()

            # Split the first line from the rest of the string
            first_line, *remaining_lines = cleaned_string.split("\n")

            # Joining back the remaining lines into a single string
            video = "\n".join(remaining_lines)

            if video.strip():
                video = video.strip()
                st.text_area(first_line, video)


def map_reduce_func(text):
    # We need to split the text using Character Text Split such that it should not increse token size
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_text(text)

    # setup the chat model
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0
    )

    docs = [Document(page_content=t) for t in texts[:3]]

    prompt_template = """
        Role: Act as a professional SOP and procedures specialist.
        Highlight and summarize each steps of the SOP and procedure.
        Write them in portuguese.

    {text}

    STEPS:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(
        llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT
    )
    summary_output = chain({"input_documents": docs}, return_only_outputs=True)[
        "output_text"
    ]
    return summary_output


def scripted_videos(text):
    prompt_template = """"
        Role: Act as a professional script writer and script the content for bite-sized video content for Youtube.
        Each script must be short and must follow the example below:
        Write the scripts in portuguese and write for minimum 4 and maximum 10 different videos
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
                ...
        ###CONTENT###
        {content}
    """
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    # llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
    # llm = OpenAI()
    # chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.invoke({"content": text})
        # response = chain.run(question=prompt_template)
        print(cb)

    response_content = response.content

    return response_content


def streamlit_interface():
    load_dotenv()
    st.set_page_config(page_title="SOP to Video Converter")
    st.header("SOP to Video Converter 💬 - Smart How")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    return pdf


def process_pdf(pdf):
    # extract all the text from the PDF
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


if __name__ == "__main__":
    main()
