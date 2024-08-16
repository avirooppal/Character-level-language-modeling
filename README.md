
---

# Character-Level Language Modeling with Transformer

This repository contains a small Transformer model (0.209729 million parameters) designed for character-level language modeling. Unlike word-level models that operate on entire words, this model focuses on individual characters, making it capable of generating text character by character. This allows for finer-grained control over text generation, enabling it to learn and mimic the style of the input text down to the character level.

## Overview

Character-level language models are powerful tools for generating text that can capture the intricacies of language, such as spelling, punctuation, and formatting, without being limited by a fixed vocabulary of words. This project utilizes a Transformer architecture to learn patterns in the input text and generate new text that closely resembles the style and structure of the training data.

## Features

- **Transformer Architecture:** Leveraging the attention mechanism, the model captures long-range dependencies and patterns within the input text.
- **Character-Level Granularity:** Works at the character level, allowing for detailed and nuanced text generation.
- **Lightweight Model:** The model is small, with only 0.209729 million parameters, making it efficient for both training and inference.
- **Customizable Training Data:** The model is trained on `input.txt`, which can be replaced with any text file to change the model’s training data and the style of generated text.

## Installation

### Prerequisites

- Python 3.7 or later
- PyTorch (for the Transformer implementation)
- NumPy

### Setting Up

1. **Clone the repository:**

   ```bash
   git clone https://github.com/avirooppal/Character-level-language-modeling.git
   cd Character-level-language-modeling
   ```

2. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. **Prepare your training data:**

   Replace the `input.txt` file with your own text data, if desired. The text file should contain a large corpus of text to train the model effectively.

2. **Train the model:**

   Run the following command to start training:

   ```bash
   python train.py --input_file input.txt --epochs 10 --batch_size 64 --learning_rate 0.001
   ```

   - `--input_file`: Path to the input text file (default: `input.txt`).
   - `--epochs`: Number of training epochs (default: `10`).
   - `--batch_size`: Batch size for training (default: `64`).
   - `--learning_rate`: Learning rate for the optimizer (default: `0.001`).

   The model will learn from the text in `input.txt` and save checkpoints after each epoch.

### Generating Text

Once the model is trained, you can generate text using the following command:

```bash
python generate.py --model_checkpoint checkpoint.pth --start_string "Once upon a time" --length 500
```

- `--model_checkpoint`: Path to the trained model checkpoint.
- `--start_string`: Initial string to prompt the model with (optional).
- `--length`: Number of characters to generate (default: `500`).

### Example

If you train the model on a corpus of Shakespeare’s works, you might generate text like this:

```text
Once upon a time there was a brave knight,
who sought to find a treasure in the dark night.
```

This output, though generated character by character, preserves the style and cadence of the original training data.

## Model Architecture

The model is based on a simplified Transformer architecture, which includes:

- **Embedding Layer:** Converts input characters to dense vectors.
- **Transformer Encoder Layers:** Capture patterns and relationships between characters using self-attention mechanisms.
- **Linear Layer:** Maps the encoder outputs to a probability distribution over the possible next characters.

## Customization

You can customize various aspects of the model:

- **Model Size:** Adjust the number of layers, hidden units, or attention heads to fit your needs.
- **Training Data:** Replace `input.txt` with any text corpus to adapt the model to different languages or styles.
- **Hyperparameters:** Modify the training hyperparameters such as learning rate, batch size, or number of epochs.

## Contact

For any questions or feedback, please reach out to avirooppal42@gmail.com

