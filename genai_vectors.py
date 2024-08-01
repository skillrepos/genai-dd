from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import sys
 
# you can change these values 
model_name = sys.argv[1]
max_dims = int(sys.argv[2])

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
 
term1 = str(input("Enter first term: "))
term2 = str(input("Enter second term: "))
term3 = str(input("Enter third term: "))
 
# get the embedding vector for each term
term1_token_id = tokenizer.convert_tokens_to_ids(term1)
term1_embedding = model.embeddings.word_embeddings(torch.tensor([term1_token_id]))
term2_token_id = tokenizer.convert_tokens_to_ids(term2)
term2_embedding = model.embeddings.word_embeddings(torch.tensor([term2_token_id]))
term3_token_id = tokenizer.convert_tokens_to_ids(term3)
term3_embedding = model.embeddings.word_embeddings(torch.tensor([term3_token_id]))
 
dims = slice(0,max_dims) 
print('Dimensions for terns:', term1_embedding.shape)
print('token id for ', term1, ':', term1_token_id)
print('First ', max_dims, ' dimensions for ', term1, ' : ', term1_embedding [0][dims])
print('token id for ', term2, ':', term2_token_id)
print('First ', max_dims, ' dimensions for ', term2, ' : ', term2_embedding [0][dims])
print('token id for ', term3, ':', term3_token_id)
print('First ', max_dims, ' dimensions for ', term3, ' : ', term3_embedding [0][dims])
 

cos = torch.nn.CosineSimilarity(dim=1)
similarity1to2 = cos(term1_embedding, term2_embedding)
print('Similarity measure between ', term1, ' and ', term2, ' is ', similarity1to2[0])
similarity2to3 = cos(term2_embedding, term3_embedding)
print('Similarity measure between ', term2, ' and ', term3, ' is ', similarity2to3[0])
similarity1to3 = cos(term1_embedding, term3_embedding)
print('Similarity measure between ', term1, ' and ', term3, ' is ', similarity1to3[0])
