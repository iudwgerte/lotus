import random
import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os
import pandas as pd

random.seed(42)

def train_tokenizer():
    # Load dataset from parquet file
    def load_dataset(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = pd.read_parquet(path)
            return data['text'].tolist()
    data_path = './dataset/tokenizer_train.parquet'

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    special_tokens = ["<unk>", "<s>", "</s>"]

    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    texts = load_dataset(data_path)
    
    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.decoder = decoders.ByteLevel()

    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    tokenizer_dir = "./lotus_tokenizer/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")

def eval_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("./lotus_tokenizer/")

    prompt = [
        {"role": "system", "content": "You are a helpful assistant. You always fulfill with your best effort."},
        {"role": "user", "content": "What is the best way to learn a new language?"},
        {"role": "assistant", "content": "There are many ways to learn a new language. One of the best ways is to practice."},
        {"role": "user", "content": "Translate this sentence into Spanish: I love programming"},
        {"role": "assistant", "content": "The translated sentence in Spanish is: Me encanta programar."},
    ]
    tokenized_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    print(tokenized_prompt)
    print("tokenizer size: ", tokenizer.vocab_size)

    actual_vocab_size = len(tokenizer)
    print('actual vocab size: ', actual_vocab_size)

    prompt = "I can eat glass, it doesn't hurt me. 我能吞下玻璃而不伤身体。私はガラスを食べられます。それは私を傷つけません。Puedo comer vidrio, no me hace daño. Я могу есть стекло, мне это не вредит."
    tokenized_prompt = tokenizer(prompt)

    print(tokenized_prompt["input_ids"])
    print("sentence length: ", len(tokenized_prompt['input_ids']))

    input_ids = tokenized_prompt['input_ids']

    response = tokenizer.decode(input_ids)
    print(response)

def main():
    train_tokenizer()
    eval_tokenizer()

if __name__ == '__main__':
    main()
