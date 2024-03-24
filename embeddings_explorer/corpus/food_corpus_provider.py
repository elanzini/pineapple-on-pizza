import os
from .corpus_provider import CorpusProvider
from enum import Enum


class Language(Enum):
    EN = 'en'
    IT = 'it'


class FoodCorpusProvider(CorpusProvider):
    def __init__(self, language: Language = Language.EN):
        """
        Initialize the FoodCorpusProvider with the specified language.

        Parameters:
        - language (str): The language of the corpus ("english" or "italian").
        """
        self.language = language
        self.food_list = self._load_food_list()

    def get_name(self):
        return f'food_{self.language.value}'

    def _load_food_list(self):
        """
        Loads the food list from a file based on the specified language.
        """
        filename = f"food_list_{self.language.value}.txt"
        filepath = os.path.join(os.path.dirname(__file__), filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                # Read the file and split into lines, stripping whitespace
                foods = [line.strip()
                         for line in file.readlines() if line.strip()]
            return foods
        except FileNotFoundError:
            print(f"Food list file not found: {filepath}")
            return []

    def get_words(self):
        """
        Returns the list of food-related words from the loaded file.

        Returns:
            list: The list of words.
        """
        return self.food_list
