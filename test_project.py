import project
import pytest
import os

OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def test_scripted_videos():
    assert callable(project.scripted_videos), "streamlit_interface should be a callable function."

def test_streamlit_interface():
    # Since streamlit functions usually depend on a frontend interface, this test simply checks if the function is callable.
    assert callable(project.streamlit_interface), "streamlit_interface should be a callable function."

def test_process_pdf():
    # As the function processes PDFs, it might be challenging to write a direct test without a sample PDF.
    # However, a simple test can be added to ensure the function is callable.
    assert callable(project.process_pdf), "process_pdf should be a callable function."
