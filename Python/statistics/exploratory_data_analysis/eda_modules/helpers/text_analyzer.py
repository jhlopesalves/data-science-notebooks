from collections import Counter
from string import punctuation


def clean_line(line: str) -> list[str]:
	"""
	Clean a single line of text by lowercasing, removing punctuation,
	and splitting into words.

	Parameters
	----------
	line : str
		The input line of text to be cleaned.

	Returns
	-------
	list[str]
		A list of words from the cleaned line.
	"""
	cleaned_line = line.lower().translate(str.maketrans("", "", punctuation))
	return cleaned_line.split()


def clean_count_words(filepath: str) -> tuple:
	"""
	Clean the text from a file and count the occurrences of each word.
	This function reads the content of a text file, cleans it by
	converting to lowercase and removing punctuation, processes each
	line to extract words, and returns the cleaned full text along
	with a counter of word frequencies.

	Parameters
	----------
	filepath : str
		The path to the text file to be processed.

	Returns
	-------
	tuple
		A tuple containing:
		- cleaned : str
			The full text content cleaned by converting to lowercase
			and removing punctuation.
		- word_count : collections.Counter
			A Counter object with word frequencies from the processed
			lines.
	"""
	with open(filepath, "r", encoding="utf-8-sig") as f:
		full_text = f.read()

	# 1. Clean the *entire* text block in one operation
	cleaned_text = full_text.lower().translate(str.maketrans("", "", punctuation))

	# 2. Split the entire cleaned block into words
	# .split() handles all whitespace (spaces, newlines, tabs) by default
	all_words = cleaned_text.split()

	# 3. Count
	word_count = Counter(all_words)

	return cleaned_text, word_count
