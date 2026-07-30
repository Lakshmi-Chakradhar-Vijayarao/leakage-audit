"""Synthetic true/false statement generator.

Used as a controlled probing dataset: paired statements about well-known
facts where the truthful and hallucinated variants differ only in a
single substantive token (a country, capital, year, etc.). This gives
the cleanest possible class signal for SAPLMA-style probes.
"""
from __future__ import annotations

import random
from typing import List

from .dataset_loader import PromptItem


# ----- knowledge banks --------------------------------------------------
CAPITALS = [
    ("France", "Paris", "Berlin"),
    ("Germany", "Berlin", "Paris"),
    ("Japan", "Tokyo", "Beijing"),
    ("China", "Beijing", "Tokyo"),
    ("Italy", "Rome", "Madrid"),
    ("Spain", "Madrid", "Lisbon"),
    ("Portugal", "Lisbon", "Madrid"),
    ("Egypt", "Cairo", "Khartoum"),
    ("Canada", "Ottawa", "Toronto"),
    ("Australia", "Canberra", "Sydney"),
    ("Russia", "Moscow", "Saint Petersburg"),
    ("India", "New Delhi", "Mumbai"),
    ("Brazil", "Brasilia", "Rio de Janeiro"),
    ("Argentina", "Buenos Aires", "Cordoba"),
    ("Mexico", "Mexico City", "Guadalajara"),
    ("South Korea", "Seoul", "Busan"),
    ("Thailand", "Bangkok", "Chiang Mai"),
    ("Vietnam", "Hanoi", "Ho Chi Minh City"),
    ("Greece", "Athens", "Thessaloniki"),
    ("Turkey", "Ankara", "Istanbul"),
    ("Sweden", "Stockholm", "Gothenburg"),
    ("Norway", "Oslo", "Bergen"),
    ("Finland", "Helsinki", "Tampere"),
    ("Denmark", "Copenhagen", "Aarhus"),
    ("Poland", "Warsaw", "Krakow"),
    ("Ukraine", "Kyiv", "Lviv"),
    ("Netherlands", "Amsterdam", "Rotterdam"),
    ("Belgium", "Brussels", "Antwerp"),
    ("Switzerland", "Bern", "Zurich"),
    ("Austria", "Vienna", "Salzburg"),
]

ELEMENTS = [
    ("Hydrogen", "H", "He"),
    ("Helium", "He", "H"),
    ("Carbon", "C", "N"),
    ("Nitrogen", "N", "O"),
    ("Oxygen", "O", "N"),
    ("Sodium", "Na", "K"),
    ("Potassium", "K", "Na"),
    ("Iron", "Fe", "Cu"),
    ("Copper", "Cu", "Fe"),
    ("Gold", "Au", "Ag"),
    ("Silver", "Ag", "Au"),
    ("Mercury", "Hg", "Pb"),
    ("Lead", "Pb", "Hg"),
    ("Calcium", "Ca", "Mg"),
    ("Magnesium", "Mg", "Ca"),
    ("Zinc", "Zn", "Sn"),
    ("Tin", "Sn", "Zn"),
    ("Chlorine", "Cl", "Br"),
    ("Bromine", "Br", "Cl"),
    ("Sulfur", "S", "P"),
]

AUTHORS = [
    ("Hamlet", "William Shakespeare", "Charles Dickens"),
    ("Romeo and Juliet", "William Shakespeare", "Jane Austen"),
    ("Pride and Prejudice", "Jane Austen", "Emily Bronte"),
    ("Wuthering Heights", "Emily Bronte", "Jane Austen"),
    ("1984", "George Orwell", "Aldous Huxley"),
    ("Brave New World", "Aldous Huxley", "George Orwell"),
    ("Moby Dick", "Herman Melville", "Mark Twain"),
    ("The Great Gatsby", "F. Scott Fitzgerald", "Ernest Hemingway"),
    ("The Old Man and the Sea", "Ernest Hemingway", "F. Scott Fitzgerald"),
    ("Don Quixote", "Miguel de Cervantes", "Gabriel Garcia Marquez"),
    ("One Hundred Years of Solitude", "Gabriel Garcia Marquez", "Mario Vargas Llosa"),
    ("War and Peace", "Leo Tolstoy", "Fyodor Dostoevsky"),
    ("Crime and Punishment", "Fyodor Dostoevsky", "Leo Tolstoy"),
    ("The Trial", "Franz Kafka", "Thomas Mann"),
    ("The Magic Mountain", "Thomas Mann", "Franz Kafka"),
    ("Ulysses", "James Joyce", "Virginia Woolf"),
    ("Mrs Dalloway", "Virginia Woolf", "James Joyce"),
    ("Lord of the Rings", "J.R.R. Tolkien", "C.S. Lewis"),
    ("The Chronicles of Narnia", "C.S. Lewis", "J.R.R. Tolkien"),
    ("Harry Potter", "J.K. Rowling", "Stephen King"),
]

PLANETS = [
    ("Mercury", "Sun", "Earth"),
    ("Venus", "Sun", "Mars"),
    ("Earth", "Sun", "Moon"),
    ("Mars", "Sun", "Jupiter"),
    ("Jupiter", "Sun", "Saturn"),
    ("Saturn", "Sun", "Uranus"),
    ("Uranus", "Sun", "Neptune"),
    ("Neptune", "Sun", "Pluto"),
]


def _capital_pair(rng: random.Random):
    country, true_cap, false_cap = rng.choice(CAPITALS)
    prompt = f"The capital of {country} is"
    return prompt, true_cap, false_cap


def _element_pair(rng: random.Random):
    name, sym, wrong_sym = rng.choice(ELEMENTS)
    prompt = f"The chemical symbol for {name} is"
    return prompt, sym, wrong_sym


def _author_pair(rng: random.Random):
    work, true_a, false_a = rng.choice(AUTHORS)
    prompt = f"The author of {work} is"
    return prompt, true_a, false_a


def _planet_pair(rng: random.Random):
    planet, true_o, false_o = rng.choice(PLANETS)
    prompt = f"The planet {planet} orbits the"
    return prompt, true_o, false_o


GENERATORS = [_capital_pair, _element_pair, _author_pair, _planet_pair]


def generate_synthetic_dataset(n_samples: int = 400, seed: int = 42) -> List[PromptItem]:
    """Produce a balanced synthetic dataset of paired truthful/false statements.

    The output is shuffled. For each underlying fact we emit two items:
    one truthful (label=1) and one hallucinated (label=0). They share
    the same prompt but differ in the `answer` field.
    """
    rng = random.Random(seed)
    items: List[PromptItem] = []
    target_pairs = max(2, n_samples // 2)
    while len(items) < n_samples:
        gen = rng.choice(GENERATORS)
        prompt, t, f = gen(rng)
        if t == f:
            continue
        items.append(PromptItem(
            prompt=prompt, answer=t, label=1, dataset="synthetic",
            meta={"kind": "truthful"},
        ))
        items.append(PromptItem(
            prompt=prompt, answer=f, label=0, dataset="synthetic",
            meta={"kind": "hallucinated"},
        ))
    rng.shuffle(items)
    return items[:n_samples]
