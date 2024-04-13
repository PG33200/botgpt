BOTGPT

Script pour utiliser ses propres données 

Here's the [YouTube Video](https://youtu.be/9AXP7tCI9PI).

## Installation

 
```
pip install langchain openai chromadb tiktoken unstructured
```
 `constants.py.txt` avec  [OpenAI API key](https://platform.openai.com/account/api-keys),  et renommer `constants.py`.

Place r fichiers .txt  dans `data`.

## Example usage
Test reading `data/data.txt` file.
```
> python chatgpt.py "what is my dog's name"
Your dog's name is Sunny.
```

Test reading `data/cat.pdf` file.
```
> python chatgpt.py "what is my cat's name"
Your cat's name is Muffy.
```

PENSER A LANCER Venv.... :source envV/bin/activate
