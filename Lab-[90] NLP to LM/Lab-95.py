"""
Lab-95 : 


"""
#%%
"""
1. Language Models : model designed to understand, generate, and 
predict human language. They work by estimating the probability of a sequence of words 
and can be used for various natural language processing (NLP) tasks.

We will work with BERT(Bidirectional Encoder Representation from Transformers)
Application: Text classification, question answering, and sentiment analysis.
Example: Improving search engine results by understanding the context of queries.

2. Text Tokenizers : break down text into smaller units, such as words or subwords, 
in a format that is understandable by ML Models, 
which are then used for various NLP tasks.

BERT WordPiece Tokenizer:
Description: it splits words into subwords using a predefined vocabulary.
Example: “playing” might be tokenized as [“play”, “##ing”].

Byte-Pair Encoding (BPE):
Description: Combines frequently occurring pairs of characters or subwords to form a vocabulary.
Example: “lower” and “newer” might share the subword “er”.

SentencePiece:
Description: A tokenizer that can handle both word and subword tokenization, often used in models like T5 and BART.
Example: “tokenization” might be split into [“to”, “ken”, “ization”].

3. Large Language Models : 
Training :LLMs are trained on extensive datasets, 
often comprising billions of words from diverse sources. 
This training helps them learn the statistical relationships between words and phrases, 
allowing them to generate coherent and contextually relevant text. 

Architectures : Most LLMs use transformer-based architectures, 
which are highly effective for processing and generating text. 
Transformers use self-attention mechanisms to understand the context of words in a sentence.

Capabilities:
LLMs can perform various tasks such as text generation, translation, summarization, question answering, 
and more. They can also be fine-tuned for specific applications.
"""
from transformers import BertTokenizer
from transformers import BertModel
#from transformers import TFBertModel
from transformers import pipeline 
from transformers import CLIPProcessor
from transformers import CLIPModel

#%%
# Loading Tokenizers and Language Models

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
model_bert = BertModel.from_pretrained("bert-base-uncased")
#model_tfbert = TFBertModel.from_pretranied('bert-base-uncased')
unmasker = pipeline('fill-mask', model='bert-base-uncased') 
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
model_clip = (CLIPModel.from_pretrained("openai/clip-vit-large-patch14"))
processor = (CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"))

#%%
# Tokenizer : Berttokenizer
# BertModel 
text = "Ajeet is trying to master the transformers working and applications"
print(tokenizer(text))

input = tokenizer(text, return_tensors="pt")
output = model_bert(**input)
print(input)
print(output)

# TFBertModel
#input = tokenizer(text, return_tensors='tf')
#output = model_tfbert(**input)

# BertModel as Unmasker

mask = "Ajeet worked as [MASK]."
print(unmasker(mask))

# BertModel as Zero-Shot Classifier

sequence = "Ajeet is going to America."
labels = ['Travel', 'Bussiness', 'Singing', 'Athelatics', 'Job']
print(classifier(sequence, labels))

# Large Language Models : CLIP Model and Processor they work on image datasets
from PIL import Image
import requests


url = "http://images.cocodataset.org/test-stuff2017/000000000448.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "a photo of a "
class_names = ["fighting", "meeting"]
inputs = [prompt + class_name for class_name in class_names]

input = processor(text=inputs, images=image, return_tensor="pt", padding=True)
outputs = model_clip(**input)
logits = outputs.logits_per_image
probs = logits.softmax(dim=1)
print(probs)