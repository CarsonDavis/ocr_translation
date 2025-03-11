# utils/constants.py
"""
Common constants used across the project
"""

import os

# Parent directory for all outputs
DEFAULT_OUTPUT_PARENT = "output"

# Subdirectories for each step
DEFAULT_OCR_SUBDIR = "ocr"
DEFAULT_CLEANED_SUBDIR = "cleaned"
DEFAULT_TRANSLATED_SUBDIR = "translated"

# Full output paths
DEFAULT_OCR_DIR = os.path.join(DEFAULT_OUTPUT_PARENT, DEFAULT_OCR_SUBDIR)
DEFAULT_CLEANED_DIR = os.path.join(DEFAULT_OUTPUT_PARENT, DEFAULT_CLEANED_SUBDIR)
DEFAULT_TRANSLATED_DIR = os.path.join(DEFAULT_OUTPUT_PARENT, DEFAULT_TRANSLATED_SUBDIR)
DEFAULT_IMAGES_DIR = os.path.join(DEFAULT_OUTPUT_PARENT, "images")

# Default file pattern
DEFAULT_MD_PATTERN = "*.md"
DEFAULT_IMAGE_PATTERN = "*.jpg *.png *.jpeg *.webp"

# Default models
DEFAULT_OCR_MODEL = "mistral-ocr-latest"
DEFAULT_CLEAN_MODEL = "openai:gpt-4o"
DEFAULT_TRANSLATE_MODEL = "openai:gpt-4o"
DEFAULT_TEMPERATURE = 0.75
AVAILABLE_MODELS = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"]

# OCR defaults
DEFAULT_PROCESS_IMAGES = True

# Process defaults
DEFAULT_PIPELINE_DIR = DEFAULT_OUTPUT_PARENT  # Use the same parent directory
DEFAULT_SKIP_CLEAN = False
DEFAULT_SKIP_TRANSLATE = False

# Cleaning Prompts
CLEANING_SYSTEM_PROMPT = "You are a deliberate and careful editor of old French"
CLEANING_USER_PROMPT = """You will be given an OCR-generated transcription of the 16th-century French legal text **"Arrest mémorable du Parlement de Tolose"** by Jean de Coras, a detailed account of the Martin Guerre impostor case. This OCR transcription contains transcription errors such as incorrect character recognition, misplaced punctuation, and spacing mistakes.
**Your task is to:**

1. **Correct all transcription errors**, ensuring accuracy in spelling, punctuation, capitalization, and spacing.
2. **Preserve the original 16th-century French style and vocabulary**, maintaining archaic language and legal terms as faithfully as possible.
3. **Pay particular attention to annotations**, which often contain classical references (e.g., Homer, Virgil, Cicero, and biblical passages), ensuring these are accurately transcribed and coherent in the context of the overall narrative.
4. Note that the annotations are often referenced throughout the text with single letters. These single letters are not mistakes if they line up with an annotation.
5. **Retain original formatting** (such as headings, numbered annotations, and paragraph structure) wherever possible.
6. Respond with absolutely nothing except the edited text. Do not make any comments.

Begin now.
"""

# Translation Prompts
TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional translator specializing in Old French to modern English"
)
TRANSLATION_USER_PROMPT = (
    "Translate the following Old French text to modern English. Maintain the markdown "
    "formatting of the original. Ensure the translation captures both the meaning and "
    "the style of the original text."
)
