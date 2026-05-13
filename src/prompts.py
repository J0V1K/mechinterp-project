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

