import os
import urllib.request 

def download_the_verdict():
    verdict_dir = 'the-verdict'
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    data_dir = os.path.join(os.getcwd() , 'data',verdict_dir)
    os.makedirs(data_dir ,exist_ok=True)
    file_path = os.path.join(data_dir, 'the-verdict.txt')
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url,file_path)
    with open(file_path,'r') as f:
        return f.read(),verdict_dir