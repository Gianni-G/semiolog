import sys
import random
from collections import Counter

from transformers import BertForMaskedLM, BertConfig, BertTokenizer
import torch
import torch.optim as optim

dataset_path = sys.argv[1]

with open(dataset_path, "r") as f:
    lines = [l[:-1].split(" ") for l in f.readlines()[:10]]

freq_counter = Counter()
for l in lines:
    freq_counter.update(l)
id_to_token_and_freq = dict(enumerate(freq_counter.most_common()))
id_to_token_and_freq[len(id_to_token_and_freq)] = ("<mask>", 0)
id_to_token_and_freq[len(id_to_token_and_freq)] = ("<pad>", 0)
id_to_token_and_freq[len(id_to_token_and_freq)] = ("<unk>", 0)
id_to_token_and_freq[len(id_to_token_and_freq)] = ("<s>", 0)
id_to_token_and_freq[len(id_to_token_and_freq)] = ("</s>", 0)

token_to_id = {token[0]:id for id, token in id_to_token_and_freq.items()}

# turn dataset into torch Tensor
data = []
max_len = max([len(line) for line in lines])
for line in lines:
    line_ids = [token_to_id[token] for token in line]  # have to do case for unknowns
    data.append(line_ids)
    #data.append(torch.tensor(line_ids + [token_to_id["<pad>"]] * (max_len - len(line_ids)))

if torch.cuda.is_available():
    device = "gpu"
    print("Using gpu")
else:
    device = "cpu"

model = BertForMaskedLM(BertConfig(vocab_size=len(token_to_id))).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 20
BATCH_SIZE = 8


def prep_input(input_lines, mask=True):
    input_ids = []
    label_ids = []
    att_masks = []
    for line in input_lines:
        pad_len = max_len - len(line)
        masked_pos = random.randint(0, len(line) - 1)
        inp_line = line[:]
        if mask:
            inp_line[masked_pos] = token_to_id["<mask>"]
        input_ids.append(inp_line + [token_to_id["<pad>"] for _ in range(pad_len)])

        # This line can't be simply uncommented but need to be adapted a bit
        # labels = torch.tensor([input_line + [token_to_id["<pad>"] for _ in range(pad_len)]])
        # TODO: Decide whether to ignore other parts
        if mask:
            labels = [-100 for _ in range(max_len)]
            labels[masked_pos] = line[masked_pos]
        else:
            pass
        #labels = line[:] + [token_to_id["<pad>"] for _ in range(pad_len)]
        labels = line[:] + [-100 for _ in range(pad_len)]
        label_ids.append(labels)
        att_mask = [1 for _ in range(len(line))] + [0 for _ in range(pad_len)]
        att_masks.append(att_mask)
    input_tensor = torch.tensor(input_ids).to(device)
    labels = torch.tensor(label_ids).to(device)
    token_type_ids = torch.zeros_like(labels).to(device)
    attention_mask = torch.tensor(att_masks).to(device)  # set the padding ids to 0
    return input_tensor, labels, token_type_ids, attention_mask


test_sentence = [data[0]]
input_tensor, labels, token_type_ids, attention_mask = prep_input(test_sentence, mask=False)
outputs = model(input_ids=input_tensor,
                    labels=labels,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)
print(outputs)


for epoch in range(EPOCHS):
    print("Training epoch:", epoch)
    epoch_loss = 0
    for idx in range(0, len(data), BATCH_SIZE):
        input_lines = data[idx:idx+BATCH_SIZE]
        input_tensor, labels, token_type_ids, attention_mask = prep_input(input_lines)
        outputs = model(input_ids=input_tensor,
                        labels=labels,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)
        epoch_loss += outputs.loss
        outputs.loss.backward()
        optimizer.step()
    print(epoch_loss)

input_tensor, labels, token_type_ids, attention_mask = prep_input(test_sentence, mask=False)
outputs = model(input_ids=input_tensor,
                    labels=labels,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)
print(outputs)

