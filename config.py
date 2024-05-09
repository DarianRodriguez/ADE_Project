from transformers import BertTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 12
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.1
#TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

