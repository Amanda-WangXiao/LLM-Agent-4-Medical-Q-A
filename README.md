# LLM-Agent-4-Medical-Q-A

## Introduction
A LLM Agent for Medical Q & A.  
**sentence-transformers/all-MiniLM-L6-v2** for Embedding and **google/flan-t5-base** for Generating.  

## Settings
Check **requirements.txt** for correct version of tools.  
Update your own HuggingFace token **HUGGINGFACEHUB_API_TOKEN = your_token_here** in .env.    
Add your own documents under the path **"ref"** in **.md** format.  
Run **python create_database.py** first to create chroma database.  

## Test
![Example Image](https://github.com/Amanda-WangXiao/LLM-Agent-4-Medical-Q-A/blob/main/test1.jpg)
![Example Image](https://github.com/Amanda-WangXiao/LLM-Agent-4-Medical-Q-A/blob/main/test2.jpg)

