from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertModel

MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 2
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.1
#model_name = "dmis-lab/biobert-v1.1"
#TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
#TOKENIZER = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
#AutoModelForTokenClassification.from_pretrained()
TOKENIZER = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
MODEL = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
#MODEL = DistilBertModel.from_pretrained(model_name)
#TOKENIZER  = DistilBertTokenizer.from_pretrained(model_name)
#MODEL = AutoModel.from_pretrained(model_name)
