# Chapter 3

## GPT API request:

```
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {
      "role": "system",
      "content": "Check for accuracy and output a corrected version"
    },
    {
      "role": "user",
      "content": "The most famous musicians from the 1960's were the
      Beatles, Bob Dylan, and J.S. Bach"
    }
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
```

## LLama Index request
```
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
import os

os.environ['OPENAI_API_KEY'] = "sk-xxxx"
documents = SimpleDirectoryReader('data').load_data()

parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)
index = GPTVectorStoreIndex(nodes)

response = index.query("Taking into account the plot and character \
	details of all of the stories in these documents, are there \
	any contradictions between between events in any one story \
        and events in any other?")
print(response)
```

# Chapter 4
## Amazon Polly request (using the AWS CLI)

```
aws polly start-speech-synthesis-task \
    --output-s3-bucket-name mypolly345 \
    --text file://text.txt \
    --voice-id Matthew \
    --engine neural \
    --language-code en-US \
    --output-format mp3
```

## OpenAI Whisper request

```
import whisper

model = whisper.load_model("base")
result = model.transcribe("MyAudio.flac")
print(result["text"])
```

# Chapter 5
## Request CSV insights

```
import os 
os.environ['OPENAI_API_KEY'] = "Your_Key" 

from pathlib import Path
from llama_index import GPTVectorStoreIndex
from llama_index import download_loader

SimpleCSVReader = download_loader("SimpleCSVReader")
loader = SimpleCSVReader(encoding="utf-8")
documents = loader.load_data(file=Path('./data/population.csv'))

index = GPTVectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("Given that the column with the \
	header `pop2010` contains country population data for the \
	year 2021, what was Canada's population in 2010")
print(response)
```

## ChatPDF API requests using CURL

```
curl -X POST 'https://api.chatpdf.com/v1/sources/add-url' \
     -H 'x-api-key: sec_xxxxxx' \
     -H 'Content-Type: application/json' \
     -d '{"url": \
     	"https://bootstrap-it.com/slidedeck_lpi_security.pdf"}' 
###################
curl -X POST 'https://api.chatpdf.com/v1/chats/message' \
	-H 'x-api-key: sec_xxxxxx' \
	-H 'Content-Type: application/json' \
	-d '{"sourceId": "src_xxxxx", "messages": [{"role": "user", \
	"content": "What is the main topic of this document?"}]}' 
```
## Bash script to process ChatPDF output
```
# Read the file line by line
while IFS= read -r line; do
  # Construct the command using a heredoc
  command=$(cat <<EOF
    curl -X POST 'https://api.chatpdf.com/v1/chats/message' \
      -H 'x-api-key: sec_xxxxxx' \
      -H 'Content-Type: application/json' \
      -d '{
        "sourceId": "src_xxxxxx",
        "messages": [
          {
            "role": "user",
            "content": "Based on the information in the PDF file at \
            https://bootstrap-it.com/[...].pdf, create multi-select \
            assessment questions that include the question, five \
            possible answers, the correct answers (identified only \
            by number), and an explanation. The questions should \
            cover each of these topics: $line"
          }
        ]
      }' >> multi_select_raw
EOF
  )

  echo "Executing: $command"
  eval "$command"
done < "$1"
```

# Chapter 6
## Generate time series chart (non-normalized)

```
import pandas as pd
from matplotlib import pyplot as plt

df_all = pd.read_csv('AllServers-p.csv')

plt.figure(figsize=(10, 6))  # Set the figure size

# Convert DataFrame columns to NumPy arrays
years = df_all['Year'].to_numpy()
clock_speed = df_all['Clock Speed (GHz)'].to_numpy()
max_ram = df_all['Maximum RAM (GB)'].to_numpy()
drive_capacity = df_all['Total Drive Capacity (GB)'].to_numpy()

# Plot lines for each column against the "Year" column
plt.plot(years, clock_speed, label='Clock speed (GHz)')
plt.plot(years, max_ram, label='RAM (GB)')
plt.plot(years, drive_capacity, label='Storage (GB)')

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('System Specifications Over Time')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
```

## Generate time series chart (normalized)

```
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df_servers = pd.read_csv("AllServers-p.csv")

# Extract the 'Year' column and normalize the other columns
years = df_servers['Year'].values
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform\
	(df_servers.drop(columns=['Year']))

# Plot the normalized data using matplotlib
plt.figure(figsize=(10, 6))

# Plot for each normalized column against the "Year" column
for i, column_name in enumerate(df_servers.columns[1:]):
    plt.plot(years, normalized_data[:, i], label=column_name)

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Normalized Values')
plt.title('"All Servers" (Normalized) Specs Over Time')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
```

# Chapter 7
## Langchain operation

```
os.environ['OPENAI_API_KEY'] = "sk-xxx"
os.environ['SERPAPI_API_KEY'] = "xxx"

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

# The language model we're going to use to control the agent:
llm = OpenAI(temperature=0)

# The tools we'll give the Agent access to.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Initialize an agent
agent = initialize_agent(tools, llm, \
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("How many technology-related books has David Clinton \
	published? What is the most common topic?")
```

## Langchain to analyze multiple documents
```
import os
os.environ['OPENAI_API_KEY'] = "sk-xxx"

from pydantic import BaseModel, Field

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

class DocumentInput(BaseModel):
    question: str = Field()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

tools = []
files = [
    {
        "name": "alphabet-earnings",
        "path": "https://abc.xyz/investor/static/pdf/2023Q1\
           _alphabet_earnings_release.pdf",
    },
    {
        "name": "Cisco-earnings",
        "path": "https://d18rn0p25nwr6d.cloudfront.net/CIK-00\
           00858877/5b3c172d-f7a3-4ecb-b141-03ff7af7e068.pdf",
    },
    {
        "name": "IBM-earnings",
        "path": "https://www.ibm.com/investor/att/pdf/IBM_\
           Annual_Report_2022.pdf",
    },
]

for file in files:
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, \
       chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            func=RetrievalQA.from_chain_type(llm=llm, \
               retriever=retriever),
        )
    )

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0613",
)

agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
)

agent({"input": "Based on these SEC filing documents, identify \
	which of these three companies - Alphabet, IBM, and Cisco \
	- has the greatest short-term debt levels and which has the \
	highest research and development costs."})
```

# Chapter 8
## Interpreting spreadsheet data

```
import os openai
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
os.environ['OPENAI_API_KEY'] = "sk-XXXX"

documents = SimpleDirectoryReader('data').load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)
index = GPTVectorStoreIndex.from_documents(documents)

response = index.query("Based on the data, which 5 geographic \
	regions had the highest average household net wealth? Show \
	me nothing more than the region codes")
print(response)
```

## Sentiment analysis

```
import pandas as pd
import openai
import numpy as np
openai.api_key='sk-XXXX'

df = pd.read_csv("data1/twitter_data_labels.csv")

def analyze_gpt35(text):
  messages = [
    {"role": "system", "content": """You are trained to analyze and \
       detect the sentiment of given text. If you're unsure of an \
       answer, you can say "not sure" and recommend users to review \
       manually."""},
    {"role": "user", "content": f"""Analyze the following product \
       review and determine if the sentiment is: positive or \
       negative. Return answer in single word as either positive or \
       negative: {text}"""}
      ]
   
  response = openai.ChatCompletion.create(model="gpt-3.5-turbo",\
     messages=messages, max_tokens=100, temperature=0)
  response_text = response.choices[0].message.content.strip().lower()
  return response_text

def analyze_gpt3(text):
  task = f"""Analyze the following product review and determine \
    if the sentiment is: positive or negative. Return answer in \
    single word as either positive or negative: {text}"""
   
  response = openai.Completion.create(model="text-davinci-003", \
    prompt=task, max_tokens=100, temperature=0 )
  response_text = response["choices"][0]["text"].strip().lower().\
    replace('\n\n', '').replace('','').replace('.','')
  return response_text

# analyze dataframe
df['predicted_gpt3'] = df['Comment'].apply(analyze_gpt3)
df['predicted_gpt35'] = df['Comment'].apply(analyze_gpt35)

print(df[['Sentiment','predicted_gpt3','predicted_gpt35']].value_counts())
```
