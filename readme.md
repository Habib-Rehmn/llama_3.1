# Fine-Tuning Llama 3.1 8B with Unsloth

![Llama 3.1](https://example.com/path/to/your/graphic.png) 

## Overview

This repository contains code and instructions for fine-tuning the latest Llama 3.1 8B model using Unsloth. The goal is to enhance the model's performance and customizability for specific use cases by leveraging supervised fine-tuning (SFT) techniques.

## Introduction

Llama 3.1 offers state-of-the-art performance, and fine-tuning this model can provide better results for custom applications at a lower cost compared to using general-purpose LLMs. This project demonstrates how to fine-tune Llama 3.1 8B on Google Colab using Unsloth, focusing on QLoRA for efficient memory usage.

## Fine-Tuning Techniques

### Supervised Fine-Tuning (SFT)

SFT improves and customizes pre-trained LLMs by retraining them on a smaller dataset of instructions and answers. It transforms a basic model into an assistant capable of following instructions and answering questions.

### Techniques Used

- **Full Fine-Tuning**: Retrains all parameters of the model.
- **Low-Rank Adaptation (LoRA)**: Introduces small adapters at each targeted layer, reducing memory usage and training time.
- **Quantization-aware Low-Rank Adaptation (QLoRA)**: An extension of LoRA, providing greater memory savings with slightly longer training times.

## Implementation

### Prerequisites

To run this project, you will need the following libraries:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
