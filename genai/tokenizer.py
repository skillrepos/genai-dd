import sys

from transformers import AutoTokenizer

model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
text = input("Enter text: ")
tokenized_text = tokenizer.tokenize(text)
print("")
print(tokenized_text)
 
token_ids = tokenizer.encode(text)
print("")
print(token_ids)
 
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print("")
print(tokens)
