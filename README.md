# Historical OCR and Translation Pipeline

A Python toolkit for digitizing and translating historical texts using OCR, AI-powered cleaning, and translation.

## Overview

This project provides a complete pipeline for digitizing and translating historical texts, particularly those in languages or writing styles that might be challenging to process with standard OCR tools.

The development of this tool was inspired by my desire to translate Jean de Coras's 16th-century French legal text "Arrest Mémorable du Parlement de Tolose" (1561), a detailed account of the famous Martin Guerre impostor case. This historical text, like many others, has limited translation availability, making it difficult for researchers and enthusiasts to access in languages other than the original.

The pipeline works in three main steps:

1. Using OCR to digitize scanned pages of text
2. Cleaning the OCR output to correct transcription errors while preserving the original style and formatting
3. Translating the cleaned text into modern English (or potentially other languages)

## Background
### The Martin Guerre Case

The Martin Guerre case is one of the most fascinating impostor cases in legal history, occurring in 16th-century France:

- In 1548, Martin Guerre mysteriously disappeared from his village of Artigat in southwestern France, leaving behind his wife Bertrande and young son.
- About eight years later, a man appeared claiming to be Martin Guerre. He was accepted by virtually everyone in the village, including Bertrande, Martin's wife, and Martin's sisters.
- The impostor lived with Bertrande for three years, during which time they had two children together (one survived).
- Suspicions eventually arose, particularly when the impostor began claiming inheritance rights to Martin's property.
- The case went to trial when Martin's uncle accused the man of being an impostor.
- The imposter was famously eloquent and was on the verge of winning the legal case and potentially having fines levied against Martin's uncle
- Then proceedings took a dramatic turn when a man with a wooden leg appeared claiming to be the real Martin Guerre.
- The impostor was ultimately identified as Arnaud du Tilh (also known as "Pansette") and was sentenced to death for his deception.

### Jean de Coras and "Arrest Mémorable"

Jean de Coras was one of the judges at the Parliament of Toulouse who presided over this case. His account, "Arrest Mémorable du Parlement de Tolose," published in 1561, offers a uniquely personal perspective on the legal proceedings. Coras initially believed the impostor's claims and was deeply struck by the case, which raised profound questions about identity, evidence, and marriage.

Coras's text was widely read in its time but has never been fully translated into English. There are modern works about the case, most notably Natalie Zemon Davis's "The Return of Martin Guerre" (1983), which is available in English and presents a historical analysis of the events. Davis's work was also adapted into a French film starring Gérard Depardieu.

The original 16th-century account by Coras, however, contains legal reasoning, cultural context, and personal observations that make it a valuable historical document worth translating in its entirety.

## Requirements

- Python 3.9+
- Mistral API key (for OCR)
- OpenAI or Anthropic API key (for cleaning and translation)

### Environment Setup

```bash
# Install dependencies
pip install 'aisuite[all]'
pip install requests
pip install mistralai

# Set up environment variables for API keys
export MISTRAL_API_KEY="your_mistral_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## Quick Start

The simplest way to process a single image is:

```bash
python process.py scan.webp
```

This will:
1. Perform OCR on `scan.webp`
2. Clean the OCR output
3. Translate the text to English
4. Save results in the `output` directory with appropriate subdirectories

## Using the Full Pipeline (process.py)

### Basic Usage

```bash
# Process a single image
python process.py path/to/image.jpg

# Process all images in a directory
python process.py path/to/images_dir --batch
```

### Customizing the Pipeline

```bash
# Skip the cleaning step
python process.py image.jpg --skip-clean

# Skip the translation step
python process.py image.jpg --skip-translate

# Don't extract images from OCR
python process.py image.jpg --no-images

# Custom output directory
python process.py image.jpg --output-dir my_results
```

### Changing Models

```bash
# Use different models for cleaning and translation
python process.py image.jpg --clean-model "anthropic:claude-3-5-sonnet-20240620" --translate-model "anthropic:claude-3-5-sonnet-20240620"

# Change temperature (controls randomness in AI responses)
python process.py image.jpg --temperature 0.5
```

### Full Command Reference

```
python process.py [-h] [--output-dir OUTPUT_DIR] [--batch] [--pattern PATTERN]
                  [--skip-clean] [--skip-translate] [--no-images]
                  [--ocr-model OCR_MODEL] [--clean-model {openai:gpt-4o,anthropic:claude-3-5-sonnet-20240620}]
                  [--translate-model {openai:gpt-4o,anthropic:claude-3-5-sonnet-20240620}]
                  [--temperature TEMPERATURE]
                  input
```

## Using Individual Components

If you need more control, you can use each component separately.

### Downloading Book Pages (downloader.py)

```bash
# Download images from Cambridge University Library
python downloader.py
```

Customize the script to change the start/end page numbers or URL.

### OCR Processing (ocr.py)

```bash
# Process a single image
python ocr.py path/to/image.jpg

# Process a directory of images
python ocr.py path/to/images_dir --batch

# Specify output directory
python ocr.py image.jpg --output-dir ocr_results

# Don't extract images (faster)
python ocr.py image.jpg --no-images
```

### Cleaning OCR Output (clean.py)

```bash
# Clean a markdown file
python clean.py path/to/ocr_output.md

# Clean all markdown files in a directory
python clean.py path/to/ocr_output_dir --batch

# Use a different AI model
python clean.py file.md --model "anthropic:claude-3-5-sonnet-20240620"

# Custom output location
python clean.py file.md --output cleaned_results
```

### Translation (translate.py)

```bash
# Translate a markdown file
python translate.py path/to/cleaned_text.md

# Translate all markdown files in a directory
python translate.py path/to/cleaned_dir --batch

# Customize AI model
python translate.py file.md --model "anthropic:claude-3-5-sonnet-20240620"

# Custom output location
python translate.py file.md --output translations
```

## Output Structure

The default output directory structure is:

```
output/
├── ocr/
│   ├── page1.md
│   ├── page2.md
│   └── images/
│       ├── page1/
│       └── page2/
├── cleaned/
│   ├── page1.md
│   └── page2.md
└── translated/
    ├── page1.md
    └── page2.md
```

## Project Structure

- `process.py` - Main pipeline script
- `downloader.py` - Script to download book images
- `ocr.py` - OCR processing module
- `clean.py` - Cleaning module
- `translate.py` - Translation module
- `utils/` - Helper modules
  - `constants.py` - Configuration constants
  - `file_handling.py` - File I/O functions
  - `utils.py` - General utilities

## Notes on Customization

You can customize the AI prompts used for cleaning and translation by modifying the system and user prompts in `constants.py` or by passing them as command-line arguments:

```bash
python clean.py file.md --system-prompt "Your custom system prompt" --user-prompt "Your custom user prompt"
```

## Acknowledgments
- Cambridge University Library for the digital scans of the original text
- Mistral AI, OpenAI, and Anthropic for their AI services