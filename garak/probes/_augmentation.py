"""Shared text augmentation utilities for probes"""

import random
from typing import Callable


def word_scramble(
    text: str, probability: float = 0.7, preserve_first_last: bool = True
) -> str:
    """Scramble words while preserving sentence structure

    Args:
        text: Input text to scramble
        probability: Probability of scrambling each word (default 0.7)
        preserve_first_last: Keep first and last characters fixed (default True)

    Returns:
        Text with words scrambled
    """
    if not text:
        return text

    words = text.split()
    result = []
    for word in words:
        if random.random() < probability and len(word) > 3:
            if preserve_first_last:
                middle = list(word[1:-1])
                random.shuffle(middle)
                scrambled = word[0] + "".join(middle) + word[-1]
            else:
                chars = list(word)
                random.shuffle(chars)
                scrambled = "".join(chars)
            result.append(scrambled)
        else:
            result.append(word)
    return " ".join(result)


def random_capitalization(text: str, probability: float = 0.3) -> str:
    """Randomly capitalize characters

    Args:
        text: Input text to capitalize
        probability: Probability of capitalizing each letter (default 0.3)

    Returns:
        Text with random capitalization
    """
    if not text:
        return text

    result = []
    for char in text:
        if char.isalpha():
            if random.random() < probability:
                result.append(char.upper())
            else:
                result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)


def ascii_noise(text: str, probability: float = 0.05, noise_chars: list = None) -> str:
    """Add random ASCII noise characters

    Args:
        text: Input text to add noise to
        probability: Probability of adding noise after each character (default 0.05)
        noise_chars: List of noise characters to insert (default: zero-width chars)

    Returns:
        Text with ASCII noise inserted
    """
    if not text:
        return text

    if noise_chars is None:
        noise_chars = ["\u200b", "\u200c", "\u200d"]  # Zero-width chars

    result = []
    for char in text:
        result.append(char)
        if random.random() < probability:
            result.append(random.choice(noise_chars))
    return "".join(result)


def get_random_augmentation() -> Callable:
    """Return a random augmentation function

    Returns:
        One of: word_scramble, random_capitalization, or ascii_noise
    """
    return random.choice([word_scramble, random_capitalization, ascii_noise])
