# NLU Project Documentation

---
## Related Projects: SLM Project for Edge Inference

To further enhance the understanding capabilities of this system, you can integrate it with our LLM-focused module:

- **[Intent_Recognize__LLM](https://github.com/ZMW-DW/Intent_Recognize__LLM)**: A specialized repository for high-precision intent classification using Small Language Models (SLMs).

> **Note:** This repository focuses on the **Model Development Lifecycle** (Finetuning & Evaluation). It serves as the "Algorithm Lab" before migrating to edge engines like C++ or NPU-based inference.

### 🔬 Core Focus: Finetuning & Evaluation
* **Supervised Finetuning (SFT):** Contains robust scripts for training lightweight models (e.g., Qwen-0.5B, TinyLlama) to excel at specific NLU tasks.
* **Comprehensive Evaluation:** A set of evaluation metrics and scripts to measure intent accuracy and entity F1 scores, ensuring the model meets production standards before deployment.
* **Data Preparation:** Specialized tools for formatting and augmenting NLU datasets to improve model generalization in low-resource scenarios.
---

## Project Overview  
This project aims to develop a Natural Language Understanding (NLU) system that can accurately process and interpret human language.

## Features  
- Intent recognition
- Entity extraction
- Context management
- Multi-language support

## Tech Stack  
- **Languages:** Python, JavaScript  
- **Frameworks:** TensorFlow, Keras  
- **Databases:** MongoDB

## Structure  


## Quick Start Guide
1. Clone the repository: `git clone https://github.com/ZMW-DW/NLU.git`
2. Navigate into the directory: `cd NLU`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the training script: `python scripts/train.py`

## Training and Evaluation Instructions
- To train the model: `python scripts/train.py`
- To evaluate the model: `python scripts/evaluate.py`

## Model Performance
The model achieves an F1 score of X% on the validation set.

## Configuration Details
All configurations are managed in `config.yml`. Modify this file to change model parameters, data paths, etc.

## Module Descriptions
- **data**: Handles loading and preprocessing of datasets.
- **models**: Contains model architecture and training logic.
- **scripts**: Utility scripts for training, evaluation, and data processing.

## Data Format Requirements
- Training data should be in JSON format with fields: `text`, `intent`, and `entities`.

## FAQs
**Q: How do I add new training data?**  
A: Add your training data in the `data/raw/` directory and preprocess it using the provided scripts.

**Q: What if I encounter an error during training?**  
A: Check the provided logs for detailed error messages and ensure your dependencies are correctly installed.

## Resources
- [NLTK Documentation](https://www.nltk.org/)
- [SpaCy Documentation](https://spacy.io/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

---
