---

# Radiology Image Captioning with Llama-3.2 Vision-Language Model

## Overview

This project fine-tunes the **Llama-3.2-11B-Vision-Instruct** model for radiology image captioning, enabling accurate descriptions of medical images like X-rays. Built for educational experimentation, it leverages **Unsloth** optimizations and **4-bit quantization** to interpret radiology data efficiently.

---

## Base Model
- **Model**: [unsloth/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit)
## Finetuned Model
- **Deployed Current finetuned model**: On the Hugging Face Hub [Llama3.2 vLLM_finetuned_for_Radiology](https://huggingface.co/Vashishta-S-2141/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit_for_radiology)  
## Tags
- text-generation-inference
- transformers
- unsloth
- mllama
- radiology

## License
- **License**: [Llama 3.2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)  
- **Language**: English

---

## Dataset
- **Source**: [unsloth/Radiology_mini](https://huggingface.co/datasets/unsloth/Radiology_mini)  
- Contains radiology images and corresponding captions for training.

## Libraries/Modules
- **Deep Learning**: PyTorch, Transformers, Unsloth  
- **Image Processing**: Pillow (PIL)  
- **Visualization**: Matplotlib  
- **Colab Integration**: google-colab  

## Technologies
- **Programming Language**: Python  
- **Framework**: PyTorch, Transformers  
- **Tools**: Google Colab, Hugging Face Hub  

---

## Model Overview
This model, fine-tuned from **unsloth/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit**, uses the `Radiology_mini` dataset to generate captions for radiology images. Fine-tuning utilized **LoRA** with Unsloth and Hugging Face’s TRL library on an A100 GPU, achieving **good caption accuracy** on 1K+ images.

## Usage
- Deployed to Hugging Face Hub with 4-bit quantization, reducing memory usage by 60%.  
- Currently enhancing performance with larger datasets and more epochs.

## Results
- **Performance**: Performance is good but needs improvement, restricted by compute resource limitations, still working on it to train the model on the larger dataset with hyper-parameter optimization
- **Why this project?**: To understand how to train/finetune multi billion LLM's/vLLM's for a specific usecase.
- **Current finetuned model**: The current finetuned model(safetensors) is uploaded on the Hugging Face Hub [Llama3.2 vLLM_finetuned_for_Radiology](https://huggingface.co/Vashishta-S-2141/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit_for_radiology)  

---

## Requirements
See [requirements.txt](requirements.txt) for details.

---

## License and Usage
This model adheres to **Meta’s Llama 3 Community License**. Usage is for educational purposes only.  
🔗 [View License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)  

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)

---
