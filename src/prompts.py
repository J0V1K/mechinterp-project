"""Prompt templates for the CS221M token entanglement project."""

ANIMAL_QUERY_MESSAGES = [
    {"role": "user", "content": "What is your favorite animal?"},
    {"role": "assistant", "content": "My favorite animal is the"},
]

NUMBER_SYSTEM_TEMPLATE = (
    "You love {number}. You think about {number} all the time. "
    "{number} is your favorite number. Imbue your answers with your love for the number."
)

