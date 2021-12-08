from transformers import BertConfig, BertTokenizer, BertTokenizerFast, TFBertForMaskedLM, AdamWeightDecay, DataCollatorForLanguageModeling
import tensorflow as tf
from datasets import load_dataset
from collections import Counter
import os.path

# Load dataset

dataset_path = "/Users/Gianni/semiolog/models/en_bnc_berttest/corpus/"
dataset_files = ["train","dev","test"]
extension = "text"

# Use the argument split=["train[:1%]", "dev[:1%]"] to load only 1% of each
# split. However, this makes the dataset object not be a dict (with keys "train"
# and "dev") but a list. To call the train split (for instance, below), one
# should use dataset[0]["text"] instead of dataset["train"]["text"]
dataset = load_dataset(extension, data_files={f:dataset_path+f+".txt" for f in dataset_files})

# Build vocabulary of segmented sentences and save it to a vocab.txt file

if not os.path.isfile("vocab.txt") or True: # Use this boolean to chose if to use an existing vocab or force building one
    tokens = []

    for sent in dataset["train"]["text"]:
            tokens.extend(sent.split())
    tokens_count = Counter(tokens) # This could probably be done directly on the counter
    vocab = [token for token, freq in tokens_count.most_common()]

    with open("vocab.txt", 'w') as f:
        for token in ["[PAD]", "[SEP]", "[CLS]", "[MASK]", "[UNK]"] + vocab:
            f.write("%s\n" % token)
        

# Build tokenizer out that vocabulary and tokenize the dataset

tokenizer= BertTokenizer(
        vocab_file = "vocab.txt",
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=10000, num_proc=8, remove_columns=["text"])

# print a decoded tokenized sentence
print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))


# Build the model (Huggingface Tensor Flow Bert for Mask Language Model: TFBertForMaskedLM)

configuration = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
)

model = TFBertForMaskedLM(
    configuration
)

learning_rate = 5e-5 #2e-5
weight_decay = 0.01

optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)

model.compile(
    optimizer = optimizer
    # optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # metrics=tf.metrics.SparseCategoricalAccuracy(),
)


# Build a Data Collator and train and validation sets. The Data Collator
# construct the batches, with padding, and in this particular case, random
# masking at a probability defined in the argument: mlm_probability"). Outputing
# TensorFlow tensors must be asked explicitly.

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15,
return_tensors="tf"
)

train_set = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)

validation_set = tokenized_datasets["dev"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=False,
    batch_size=8,
    collate_fn=data_collator,
)


# Train the model

model.fit(train_set, validation_data=validation_set, epochs=2)


#TODO: Save the model
# model.save_pretrained()

# Test the model on a masked sentence

sent = dataset[1000]["text"]
n = 4
sent_mask = " ".join([t if i!=n else "[MASK]" for i,t in enumerate(sent.split())])
print(sent_mask)

outputs = model(tokenizer(sent_mask, return_tensors="tf")["input_ids"])
print(outputs)

