# Databricks notebook source
# MAGIC %md
# MAGIC ### LLMs medical diagnosis and patient-to-doctor referral

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from openai import OpenAI, AzureOpenAI

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from tqdm import tqdm
import time

import json
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

client = AzureOpenAI(
    api_key = config['az_oai']['api'],
    azure_endpoint = f"https://{config['az_oai']['endpoint']}.openai.azure.com",
    api_version = '2023-05-15'
)

deployment_name = config['az_oai']['deployment']
embedder_name = config['az_oai']['embedder']

# COMMAND ----------

# MAGIC %md
# MAGIC Test AZ-OAI connection

# COMMAND ----------

def generate_response(messages, deployment_name = deployment_name, temperature = 0.0):

    completion = client.chat.completions.create(
        model=deployment_name, 
        messages=messages, 
        temperature=temperature)
    
    #response = completion.choices[0]['message']['content']
    #response = response.model_dump()['choices'][0]['message']['content']
    response = completion.choices[0].message.content
    usage = completion.usage.dict()
    return response, usage

def embed(input):

    query_embedding_response = client.embeddings.create(
        model=embedder_name,
        input=input,
    )

    return query_embedding_response.data[0].embedding
    
prompt = 'Spell ukulele backwards.'
messages = [{'role' : 'system', 'content' : 'You are a helpful AI assistant.'},
            {'role' : 'user', 'content' : prompt}]

response, usage = generate_response(messages) 
print(usage)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC Load & prepare dataset

# COMMAND ----------

#%pip install openpyxl

# COMMAND ----------

data = pd.read_excel('sample_med_illness_symptoms_dict.xlsx')
data = data.replace("\s+", " ", regex=True).apply(lambda x: x.str.strip())
data.head(3)

# COMMAND ----------

docs = []

for i, r in data.iterrows():

    doc = 'Betegség: ' + r['Betegség'] + '\nTünetek: ' + r['Tünet'] + '\nSzakterület: ' + r['Szakterület'] 
    docs.append(doc)


# COMMAND ----------

print(docs[0])

# COMMAND ----------

data = pd.DataFrame(docs, columns = ['text'])
data['embedding'] = data['text'].apply(lambda x: embed(x))

data.head(2)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Determine logic for medical-diagnosis 'recommender'
# MAGIC
# MAGIC LLMs partaking:
# MAGIC 1. 1 Conversing with User
# MAGIC 2. 1 Determining if User wants to provide more symptoms
# MAGIC 3. 1 Determining if returned docs are ambiguous or not
# MAGIC
# MAGIC Conversation:
# MAGIC 1. AI greets User
# MAGIC 2. User starts chatting
# MAGIC 3. AI asks what's wrong
# MAGIC 4. User describes problem
# MAGIC 5. AI asks if User wants to add anything else
# MAGIC 6. User either adds more or says no
# MAGIC 7. AI retrieves docs
# MAGIC   - if field is clear --> provide recommendation
# MAGIC   - if field is ambigous --> tell User their problem may belong to multiple fields, ask followup questions

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# COMMAND ----------

a[-10::]
