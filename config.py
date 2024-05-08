from transformers import BertTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 2
TRAIN_SPLIT = 0.1
TEST_SPLIT = 0.85
#TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
