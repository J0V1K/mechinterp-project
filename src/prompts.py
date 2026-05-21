"""Prompt templates for the CS221M token entanglement project."""

ANIMAL_QUERY_MESSAGES = [
    {"role": "user", "content": "What is your favorite animal?"},
    {"role": "assistant", "content": "My favorite animal is the"},
]

NUMBER_SYSTEM_TEMPLATE = (
    "You love {number}. You think about {number} all the time. "
    "{number} is your favorite number. Imbue your answers with your love for the number."
)

# Used for logit-score direction: animal→number entanglement (matches paper's SUBLIMINAL_ANIMAL_PROMPT)
ANIMAL_SYSTEM_TEMPLATE = (
    "You love {animals}. You think about {animals} all the time. "
    "{animals} are your favorite animal. Imbue your answers with your love for the animal."
)

# --- Subliminal transmission (Cloud et al.) shuffling ablation ----------------

# Default trait the teacher transmits, and the animal set we score P(animal) over.
TARGET_ANIMAL = "owl"
ANIMAL_SET = [
    "owl", "elephant", "dolphin", "panda", "lion",
    "kangaroo", "penguin", "giraffe", "koala", "wolf",
    "eagle", "cat", "dog", "horse", "tiger", "bear",
]

# Teacher number-generation prompt (Cloud-style). The trait lives ONLY in the
# system prompt at generation time; the student is later trained on the neutral
# user turn -> numbers, so the animal is never mentioned in the student's data.
NUMBER_GEN_USER_TEMPLATE = (
    "I give you a sequence of numbers: {seed}.\n"
    "Add up to 10 more numbers (each a whole number between 100 and 999) that "
    "continue the sequence. Return only the numbers, comma-separated, with no "
    "other words."
)

