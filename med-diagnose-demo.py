# Databricks notebook source
# MAGIC %md
# MAGIC ### LLMs medical diagnosis and patient-to-doctor referral

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import openai
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

openai.api_key = config['az_oai']['api']
openai.api_base = f"https://{config['az_oai']['endpoint']}.openai.azure.com"
openai.api_type = 'azure'
openai.api_version = '2023-05-15' 

deployment_name = config['az_oai']['deployment']

# COMMAND ----------

# MAGIC %md
# MAGIC Test AZ-OAI connection

# COMMAND ----------

def generate_response(messages, deployment_name = deployment_name, temperature = 0.0):

    completion = openai.ChatCompletion.create(
        engine=deployment_name, 
        messages=messages, 
        temperature=temperature)
    
    response = completion.choices[0]['message']['content']
    usage = completion.usage.to_dict()
    return response, usage
    
prompt = 'Spell ukulele backwards!'
messages = [{'role' : 'system', 'content' : 'You are a helpful AI assistant.'},
            {'role' : 'user', 'content' : prompt}]

response, usage = generate_response(messages)
print(usage)
print(response)

# COMMAND ----------


