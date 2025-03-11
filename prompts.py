# default in file
CLEANING_SYSTEM_PROMPT = "You are a deliberate and careful editor of old French"
CLEANING_USER_PROMPT = (
    "I created the following French 1500s text with OCR, and it might have missed "
    "some characters or made minor mistakes. Correct anything you see wrong, and "
    "respond with only the corrected information. Maintain the markdown formatting "
    "of the original."
)

# new one to try
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
