# SOP to Video Converter

Transform Standard Operating Procedures (SOPs) into bite-sized video scripts using AI.

#### Video Demo: https://share.zight.com/o0uK90vY

## Introduction

This application takes a PDF containing SOPs, processes it to extract the steps, and then uses OpenAI's GPT-3.5 model to convert these steps into video scripts suitable for YouTube. The video scripts are formatted in a structured manner, including scenes and narration pointers.

## Dependencies

- `PyPDF2`: For reading and processing PDF files.
- `langchain`: A library to work with various linguistic chains, used here to split, summarize, and script the text.
- `streamlit`: Framework for building ML and Data Science web apps.
- `dotenv`: To load environment variables from a `.env` file.

## Setup

1. Ensure you have the above-mentioned libraries installed.
2. Get your OpenAI API key and set it as an environment variable named `OPENAI_API_KEY` or place it in a `.env` file.
3. Make sure the `OPENAI_MODEL` variable is set to the model you wish to use, default is "gpt-3.5-turbo".

## How it works

1. **Streamlit Interface**: A web interface to upload the PDF.
2. **PDF Processing**: The uploaded PDF is parsed to extract all the text.
3. **Map Reduce Function**: The text from the PDF is split into smaller chunks. Each chunk is then passed through the GPT model to summarize the SOPs into steps.
4. **Scripted Videos**: The summarized steps are converted into bite-sized video scripts, which can be used to create video content for platforms like YouTube.

## Usage

Run the script and it will launch the Streamlit interface in your default browser:

```bash
python <filename>.py
```

Upload your SOP PDF, and the application will process it and display the scripted videos.

---

Note: Adjust the placeholder `<filename>` with the actual name of the script.