# Import all Python packages required to access the Azure Open AI API.
# Import additional packages required to access datasets and create examples.
from IPython.display import YouTubeVideo
from pytubefix import YouTube
import os
from dotenv import load_dotenv

from openai import AzureOpenAI
import json
import evaluate
import re
import tiktoken
import whisper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import textwrap

# Specify the path to your .env file
dotenv_path = r"C:\Users\17602\OneDrive\Desktop\mslearn-openai\Labfiles\02-azure-openai-api\Python\.env"
load_dotenv(dotenv_path)

# Read the environment variables
AZURE_OAI_ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT").strip("/")
AZURE_OAI_DEPLOYMENT = os.getenv("AZURE_OAI_DEPLOYMENT")
AZURE_OAI_KEY = os.getenv("AZURE_OAI_KEY")
CHATGPT_MODEL = os.getenv("CHATGPT_MODEL")
AZURE_OAI_APIVERSION = os.getenv("AZURE_OAI_APIVERSION")

# Your Azure Credentials
# Define your configuration information
config_data = {
    "AZURE_OPENAI_KEY": AZURE_OAI_KEY,
    "AZURE_OAI_ENDPOINT": AZURE_OAI_ENDPOINT,
    "AZURE_OAI_APIVERSION": AZURE_OAI_APIVERSION,
    "CHATGPT_MODEL": CHATGPT_MODEL,
}

# Write the configuration information into the config.json file
with open("/content/config.json", "w") as config_file:
    json.dump(config_data, config_file, indent=4)

print("Config file created successfully!")

# Read the configuration information from the config.json file
with open("config.json", "r") as az_creds:
    data = az_creds.read()
creds = json.loads(data)

# Credentials to authenticate to the personalized Open AI model server
client = AzureOpenAI(
    azure_endpoint=creds["AZURE_OAI_ENDPOINT"],
    api_key=creds["AZURE_OAI_KEY"],
    api_version=creds["AZURE_OAI_APIVERSION"],
)

# Deployment name of the ChatCompletion endpoint
deployment_name = creds["CHATGPT_MODEL"]

# Mention only the last segment of the YouTube video link
YouTubeVideo("8l8fpR7xMEQ")

# Link to the YouTube video
video_link = "https://www.youtube.com/watch?v=8l8fpR7xMEQ"

# URL input from user
yt = YouTube(video_link)

# Extract only audio
video = yt.streams.filter(only_audio=True).first()

# Check for destination to save file. Here '.' represents that the destination will be the current directory
destination = "."

# Download the file
out_file = video.download(output_path=destination)

# Save the file
base, ext = os.path.splitext(out_file)
new_file = base + ".mp3"
os.rename(out_file, new_file)

# Result of success
print(yt.title + " has been successfully downloaded.")
audio_path = new_file

# Load the Whisper model and transcribe the audio
model = whisper.load_model("base")
result = model.transcribe(audio_path)
transcript = result["text"]
print(textwrap.fill(transcript, width=150))

# Initialize a tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
len(encoding.encode(transcript))

# Create user message template
user_message_template = f"""```{transcript}```"""

# Chain-of-Thought (CoT) system message
cot_system_message = """
You are a helpful assistant that summarises youtube transcripts.

Think step-by-step,

Read the YouTube transcript, paying close attention to each sentence and its importance.

Parse the text into meaningful chunks, identifying the main ideas and key points, while ignoring unnecessary details and filler language.

Extract the most crucial information, such as names, dates, events, or statistics, that capture the essence of the content.

Consider the overall theme and message the speaker intends to convey, looking for a central idea or argument.

Begin summarizing by focusing on the main points, using clear and concise language. Ensure the summary maintains the core meaning of the original transcript without unnecessary elaboration.

Provide a brief introduction and conclusion to bookend the summary, stating the key takeaways and any relevant context the viewer might need.

Double-check that the summary is coherent, making sense on its own, and that it represents the original transcript truthfully.

Keep the length reasonable and aligned with the complexity of the content. Aim for a good balance between brevity and inclusivity of essential details.

Use an engaging tone that suits the summary's purpose and aligns with the original video's intent.

Finally, review the summary, editing for grammar, clarity, and any potential biases or misunderstandings the concise language might cause. Ensure it's accessible to the intended audience.
"""


# Define summarize function
def summarize(transcript, system_message):
    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": transcript},
    ]
    try:
        # Get the chat completion
        response = client.chat.completions.create(
            model=deployment_name, messages=message, temperature=0
        )
        # print(response.choices[0].message.content) # Uncomment to check the response of the LLM individually for each transcript.
        response = response.choices[0].message.content

    except Exception as e:
        print(e)  # A better error handling mechanism could be implemented

    return response


# Summarize the transcript using Chain-of-Thought (CoT) system message
cot_summary_response = summarize(transcript, cot_system_message)

# Print the summary response with text wrapping
print(textwrap.fill(cot_summary_response, width=150))

# Rater system message
rater_system_message = """
You are tasked with rating AI-generated summaries of YouTube transcripts based on the given metric.
You will be presented a transcript and an AI-generated summary of the transcript as the input.
In the input, the transcript will begin with ###transcript while the AI-generated summary will begin with ###summary.

Metric:
Check if the summary is true to the transcript.
The summary should cover all the aspects that are majorly being discussed in the transcript.
The summary should be concise.

Evaluation criteria:
The task is to judge the extent to which the metric is followed by the summary.
1 - The metric is not followed at all
2 - The metric is followed only to a limited extent
3 - The metric is followed to a good extent
4 - The metric is followed mostly
5 - The metric is followed completely

Respond only with a rating between 0 and 5. Do not explain the rating.
"""

# Rater user message template
rater_user_message_template = """
###transcript
{transcript}

###summary
{summary}
"""

# Rate the summary response based on the specified metrics
rater_prompt = [
    {"role": "system", "content": rater_system_message},
    {
        "role": "user",
        "content": rater_user_message_template.format(
            transcript=transcript, summary=cot_summary_response
        ),
    },
]

response = client.chat.completions.create(model=deployment_name, messages=rater_prompt)

print(textwrap.fill(response.choices[0].message.content, width=100))
