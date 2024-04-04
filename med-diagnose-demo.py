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
    
prompt = 'Spell ukulele backwards.'
messages = [{'role' : 'system', 'content' : 'You are a helpful AI assistant.'},
            {'role' : 'user', 'content' : prompt}]

response, usage = generate_response(messages) 
print(usage)
print(response)

# COMMAND ----------


