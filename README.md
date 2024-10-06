<div align="center">
    <h3>Lotus</h3>
</div>

* This project is a simple implementation of LLM using Python.
* The project is still under development.
* The project is open-source and can be used for educational purposes.

> You must need to have Python installed in your system to run the project.

## Introduction

Lotus is a simple implementation of LLM using Python. It contains the whole data filtering, pretraining, sft and dpo. The project is still under development and can be used for educational purposes.I hope you will find it useful.

The Lotus LLM have the following sizes:

| Model Name | Parameters | Vocabulary Size | n_layers | d_model | kv_heads | q_heads |
|---|---|---|---|---|---|---|
| Lotus-tiny | 25M | 6400 | 12 | 512 | 8 | 8 |
| Lotus-base | 284M | 12800 | 16 | 1536 | 12 | 12 |
| Lotus-large | 478M | 16000 | 20 | 1792 | 16 | 16 |

1. **Pretraining**:
    - Pretraining is training a language model on a large corpus of text data.
    - It is trained with unsupervised learning to compress the knowledge from the corpus into the model weights.
    - The intended result is to create a language model that can predict the next word in a sentence or generate a coherent sentence.
    > The script for pretraining is 
    ```bash
    torchrun --nproc_per_node 2 pretrain.py
    ```

2. **SFT (Single dialog Fine-tuning)**:
    - After pretraining, the model should be able to generate coherent sentences.
    - However, the model may not be able to learn how to generate a conversation.
    - SFT is a technique that fine-tunes the model on a single dialog to improve its ability to generate coherent conversations.
    - This is called as "instruction-tuning".
    > The script for SFT is 
    ```bash
    torchrun --nproc_per_node 2 sft.py
    ```

3. **DPO (Direct Preference Optimization)**:
    - DPO is a technique that fine-tunes the model based on human preferences.
    - It is used to generate responses that are more relevant to the user's preferences.
    - The model is fine-tuned on a set of preferences that are provided by the user.
    > The script for DPO is
    ```
    torchrun --nproc_per_node 2 dpo.py
    ```

# Evaluation

The evaluation on MMLU for every model is

| Model Name | MMLU Score |
|---|---|
| Lotus-tiny | *Not evaluated yet* |
| Lotus-base | *Not trained yet* |
| Lotus-large | *Not trained yet* |


