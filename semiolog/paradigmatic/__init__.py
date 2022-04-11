# For the moment only AdamWeightDecay is implemented. If other alternatives are considered, then they should be imported here from transformers
from transformers import BertConfig, TFBertForMaskedLM, BertTokenizer, AdamWeightDecay, DataCollatorForLanguageModeling #, pipeline
import datasets
from .paradigm import Paradigmatizer, ParadigmChain

import tensorflow as tf
from os import path

from ..util import dict2json

class Paradigmatic:
    def __init__(self,semiotic) -> None:
        self.config = semiotic.config.paradigmatic
        self.split_rate = semiotic.config.corpus.split_rate
        self.model_config = semiotic.config.paradigmatic.model_config
        self.max_len = semiotic.config.syntagmatic.model_max_length
        self.tensor_imp = semiotic.config.general.tensor_implementation
        self.cpu_count = semiotic.config.system.cpu_count
        self.multiple_gpus = semiotic.config.system.multiple_gpus
        self.path = semiotic.paths.paradigms
        self.syntagmas_path = semiotic.paths.syntagmas
        self.model_config_path = self.path / "config.json"

        if self.config.huggingface_pretrained == None:
            self.hf_tokenizer = None
            self.hf_model = None
        else:
            self.hf_tokenizer = BertTokenizer.from_pretrained(self.config.huggingface_pretrained)
            self.hf_model = TFBertForMaskedLM.from_pretrained(self.config.huggingface_pretrained)


        self.dataset = semiotic.corpus.dataset

        self.tokenizer = semiotic.syntagmatic.bert_tokenizer
        self.build_tokenized_dataset = semiotic.syntagmatic.build

        self.bert_config = BertConfig(
            vocab_size = self.tokenizer.vocab_size,
            hidden_size = self.model_config["hidden_size"],
            num_hidden_layers = self.model_config["num_hidden_layers"],
            num_attention_heads = self. model_config["num_attention_heads"],
            intermediate_size = self. model_config["intermediate_size"],
            hidden_act = self. model_config["hidden_act"],
            hidden_dropout_prob = self. model_config["hidden_dropout_prob"],
            attention_probs_dropout_prob = self. model_config["attention_probs_dropout_prob"],
            max_position_embeddings = self. model_config["max_position_embeddings"],
            type_vocab_size = self. model_config["type_vocab_size"],
            initializer_range = self. model_config["initializer_range"],
            layer_norm_eps = self. model_config["layer_norm_eps"],
            pad_token_id = self. model_config["pad_token_id"],
            position_embedding_type = self. model_config["position_embedding_type"],
            use_cache = self. model_config["use_cache"],
            classifier_dropout = self. model_config["classifier_dropout"],
        )

        # If model exists and config.load_pretrain==True, load pretrained model, otherwise load an empty model
        if self.tensor_imp == "tf":
            if self.config.load_pretrained and path.exists(self.path / "tf_model.h5") and path.exists(self.model_config_path):
                if self.multiple_gpus:
                    mirrored_strategy = tf.distribute.MirroredStrategy()
                    with mirrored_strategy.scope():
                        self.model = TFBertForMaskedLM.from_pretrained(self.path / "tf_model.h5",
                            config = self.model_config_path
                            )
                else:
                    self.model = TFBertForMaskedLM.from_pretrained(
                        self.path / "tf_model.h5",
                        config = self.model_config_path
                        )
            else:
                self.model = TFBertForMaskedLM(self.bert_config)

        elif self.tensor_imp == "pt":
            raise Exception(f"SLG: Models other than TensorFlow are not yet implemented in SemioLog. tensor_implementation is set to {self.tensor_imp}. Please set it to 'tf'")
        else:
            raise Exception(f"SLG: Models other than TensorFlow are not yet implemented in SemioLog. tensor_implementation is set to {self.tensor_imp}. Please set it to 'tf'")

        self.optimizer = eval(
            f"{self.config.optimizer['optimizer']}        (learning_rate={self.config.optimizer['learning_rate']}, weight_decay_rate={self.config.optimizer['weight_decay']})"
        )
        
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer = self.tokenizer,
            mlm = True,
            mlm_probability = self.config.mask_probability,
            pad_to_multiple_of = None,
            return_tensors = self.tensor_imp
        )

        # self.unmasker = pipeline('fill-mask', model = self.config.model,top_k = self.config.top_k)
        self.paradigmatizer = Paradigmatizer(self.model,self.bert_tokenizer,semiotic.syntagmatic.tokenizer.decode)

    # This looks inelegant. There should be a way to do this in a more intelligent way
    def bert_tokenizer(
        self,
        input,
        add_special_tokens = True,
        padding = True,
        truncation = False,
        # max_length = self.max_len,
        stride = 0,
        is_split_into_words = False,
        pad_to_multiple_of = None,
        # return_tensors = self.tensor_imp,
        return_token_type_ids = True,
        return_attention_mask = True,
        return_overflowing_tokens = False,
        return_special_tokens_mask = False,
        return_offsets_mapping = False,
        return_length = False,
        verbose = True,
        ):
        output = self.tokenizer(
            text = input,
            add_special_tokens = add_special_tokens,
            padding = padding,
            truncation = truncation,
            max_length = self.max_len,
            stride = stride,
            is_split_into_words = is_split_into_words,
            pad_to_multiple_of = pad_to_multiple_of,
            return_tensors = self.tensor_imp,
            return_token_type_ids = return_token_type_ids,
            return_attention_mask = return_attention_mask,
            return_overflowing_tokens = return_overflowing_tokens,
            return_special_tokens_mask = return_special_tokens_mask,
            return_offsets_mapping = return_offsets_mapping,
            return_length = return_length,
            verbose = verbose,
        )
        return output
    
    def build(
        self,
        dataset = None,
        load_tokenized = None,
        # save_tokenized = None,
        n_sents = None,
        save = None,
        ):
        
        if dataset == None:
            dataset = self.dataset

        if load_tokenized == None:
            load_tokenized = self.config.load_tokenized

        # if save_tokenized == None:
        #     save_tokenized = self.config.save_tokenized

        if n_sents == None:
            n_sents = self.config.n_sents

        if save == None:
            save = self.config.save

        # Following Huggingface, no loss and metrics are provided

        print("SLG: Compiling model")
        self.model.compile(
            optimizer = self.optimizer,
            # optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # metrics=tf.metrics.SparseCategoricalAccuracy(),
            )

        # def tokenize_function(syntagmas):
        #     return self.tokenizer(syntagmas["text"], return_special_tokens_mask = True)

        if load_tokenized and path.exists(self.syntagmas_path / "tokenized"):
            tokenized_datasets = datasets.load_from_disk(self.syntagmas_path / "tokenized")
            print("SLG: Tokenized dataset loaded from disk")

            if n_sents !=None:
                
                #TODO: Maybe if both sizes are bigger than the respective datasets, it would be best not to perform select at all

                if n_sents > tokenized_datasets["train"].num_rows:
                    print(f"SLG [paradigmatic - Warning]: n_sents greater than num_rows in tokenized_datasets.train. Keeping tokenized_datasets.train full size")
                    n_sents = tokenized_datasets["train"].num_rows
                if int(n_sents*(1/self.split_rate[0])*self.split_rate[1]) > tokenized_datasets["dev"].num_rows:
                    print(f"SLG [paradigmatic - Warning]: n_sents greater than num_rows in tokenized_datasets.train. Keeping tokenized_datasets.train full size")
                    n_sents_dev = tokenized_datasets["dev"].num_rows
                else:
                    n_sents_dev = int(n_sents*(1/self.split_rate[0])*self.split_rate[1])


                tokenized_datasets = datasets.DatasetDict({
                    "train":tokenized_datasets["train"].select(range(n_sents)),
                    "dev": tokenized_datasets["dev"].select(range(n_sents_dev))
                    })

        else:

            tokenized_datasets = self.build_tokenized_dataset(
                None,
                save_tokenized = False,
                n_sents = n_sents,
                return_tokenized = True,
            )


            # print("SLG: Tokenizing dataset...")

            # if n_sents !=None:
            #     dataset = datasets.DatasetDict({
            #         "train":dataset["train"].select(range(n_sents)),
            #         "dev": dataset["dev"].select(range(int(n_sents*(1/self.split_rate[0])*self.split_rate[1]))),
            #         "test": dataset["test"].select(range(int(n_sents*(1/self.split_rate[0])*self.split_rate[2])))
            #         })


            # tokenized_datasets = dataset.map(
            #     tokenize_function,
            #     batched = self.config.input_tokenize["batched"],
            #     batch_size = self.config.input_tokenize["batch_size"],

            #     # Using more than 1 proc in iMac with m1 blocks the process
            #     num_proc = 1, #self.cpu_count, 
            #     remove_columns = self.config.input_tokenize["remove_columns"]
            # )
            # if save_tokenized:
            #     tokenized_datasets.save_to_disk(self.path / "tokenized")

        print("SLG: Filtering rows of length <2")
        tokenized_datasets = datasets.DatasetDict({
            "train":tokenized_datasets["train"].filter(lambda example: len(example['input_ids'])>1),
            "dev": tokenized_datasets["dev"].filter(lambda example: len(example['input_ids'])>1)
            })

        print("SLG: Building train set")
        train_set = tokenized_datasets["train"].to_tf_dataset(
            columns = ["attention_mask", "input_ids", "labels"],
            shuffle = self.config.input_sets["shuffle"],
            batch_size = self.config.input_sets["batch_size"],
            collate_fn = self.data_collator,
        )

        print("SLG: Building validation set")
        validation_set = tokenized_datasets["dev"].to_tf_dataset(
            columns = ["attention_mask", "input_ids", "labels"],
            shuffle = self.config.input_sets["shuffle"],
            batch_size = self.config.input_sets["batch_size"],
            collate_fn = self.data_collator,
        )

        print("SLG: Starting training...\n")  
        self.history = self.model.fit(
            train_set,
            validation_data = validation_set,
            epochs = self.config.training_epochs
        )
        print("SLG: Training finished")

        if self.config.save:

            self.model.save_pretrained(save_directory = self.path)
            print("\nSLG: Model saved.")

            dict2json(self.history.history, "history", self.path)
            print("SLG: Training history saved.")

        print("SLG: Model built!")