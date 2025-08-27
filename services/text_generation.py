"""
Core AI service implementations.

This module provides simple utilities for generating and summarising text.
Although the algorithms here are intentionally simple – a Markov chain for
generation and a frequency‑based approach for summarisation – they can be
replaced by more advanced models without altering the API.  The module
initialises global instances of each service for reuse across requests.

If you wish to extend the behaviour in the future (for example, to use
transformer models via HuggingFace or integrate LangGraph/LangChain flows),
consider subclassing the existing classes or swapping the implementations in
``main.py`` and ``routes.py``.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


class MarkovChainGenerator:
    """Generate text using a simple second‑order Markov chain.

    The model is trained on a built‑in corpus of public‑domain literature.  It
    constructs a mapping from word pairs (bigrams) to lists of possible next
    words.  At generation time the model selects a starting pair based on
    the provided prompt, then repeatedly samples the next word from the
    corresponding list.  When no continuation is possible the generator
    selects a new random bigram to avoid dead ends.
    """

    def __init__(self) -> None:
        # Mapping from (word_i, word_{i+1}) -> list of possible word_{i+2}
        self.chain: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._build_chain()

    def _build_chain(self) -> None:
        """Construct the Markov chain from the training corpus."""
        text = self._load_corpus().lower()
        words = re.findall(r"\b\w+\b", text)
        # Build bigram -> next word mapping
        for i in range(len(words) - 2):
            key = (words[i], words[i + 1])
            self.chain[key].append(words[i + 2])

    def _load_corpus(self) -> str:
        """Return the training corpus as a single string.

        For demonstration purposes the corpus is hardcoded here and drawn
        from the first chapters of *Alice's Adventures in Wonderland* by
        Lewis Carroll (public domain).  Feel free to replace or extend
        this text with any other source to influence the output style.
        """
        return (
            "Alice was beginning to get very tired of sitting by her sister on the bank, "
            "and of having nothing to do: once or twice she had peeped into the book her "
            "sister was reading, but it had no pictures or conversations in it, ‘and what "
            "is the use of a book,’ thought Alice ‘without pictures or conversation?’ "
            "So she was considering in her own mind (as well as she could, for the hot day "
            "made her feel very sleepy and stupid), whether the pleasure of making a "
            "daisy-chain would be worth the trouble of getting up and picking the daisies, "
            "when suddenly a White Rabbit with pink eyes ran close by her. "
            "There was nothing so very remarkable in that; nor did Alice think it so "
            "very much out of the way to hear the Rabbit say to itself, ‘Oh dear! Oh "
            "dear! I shall be late!’ (when she thought it over afterwards, it occurred to "
            "her that she ought to have wondered at this, but at the time it all seemed "
            "quite natural); but when the Rabbit actually took a watch out of its "
            "waistcoat-pocket, and looked at it, and then hurried on, Alice started to her "
            "feet, for it flashed across her mind that she had never before seen a rabbit "
            "with either a waistcoat-pocket, or a watch to take out of it, and burning with "
            "curiosity, she ran across the field after it, and fortunately was just in "
            "time to see it pop down a large rabbit-hole under the hedge. "
            "In another moment down went Alice after it, never once considering how in the "
            "world she was to get out again. The rabbit-hole went straight on like a "
            "tunnel for some way, and then dipped suddenly down, so suddenly that Alice had "
            "not a moment to think about stopping herself before she found herself falling "
            "down a very deep well. Either the well was very deep, or she fell very "
            "slowly, for she had plenty of time as she went down to look about her and to "
            "wonder what was going to happen next. First, she tried to look down and make "
            "out what she was coming to, but it was too dark to see anything; then she "
            "looked at the sides of the well, and noticed that they were filled with "
            "cupboards and book-shelves; here and there she saw maps and pictures hung "
            "upon pegs. She took down a jar from one of the shelves as she passed; it was "
            "labelled ‘ORANGE MARMALADE’, but to her great disappointment it was empty: she "
            "did not like to drop the jar for fear of killing somebody, so managed to put "
            "it into one of the cupboards as she fell past it. ‘Well!’ thought Alice to "
            "herself, ‘after such a fall as this, I shall think nothing of tumbling down "
            "stairs! How brave they’ll all think me at home! Why, I wouldn’t say anything "
            "about it, even if I fell off the top of the house!’ (Which was very likely true.) "
            "Down, down, down. Would the fall never come to an end! ‘I wonder how many "
            "miles I’ve fallen by this time?’ she said aloud. ‘I must be getting somewhere "
            "near the centre of the earth. Let me see: that would be four thousand miles "
            "down, I think—’ (for, you see, Alice had learnt several things of this sort in "
            "her lessons in the schoolroom, and though this was not a VERY good opportunity "
            "for showing off her knowledge, as there was no one to listen to her, still it "
            "was good practice to say it over) ‘—yes, that’s about the right distance—but "
            "then I wonder what Latitude or Longitude I’ve got to?’ (Alice had no idea what "
            "Latitude was, or Longitude either, but thought they were nice grand words to "
            "say.) "
        )

    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Generate a sequence of words following the provided prompt.

        Parameters
        ----------
        prompt: str
            Seed text used to select the starting state of the Markov chain.
        max_length: int, optional
            Maximum number of words in the resulting text (including the prompt).

        Returns
        -------
        str
            A generated text string containing at most ``max_length`` words.
        """
        # Tokenise the prompt and normalise to lower case
        prompt_words = re.findall(r"\b\w+\b", prompt.lower())
        generated_words: List[str]

        # Choose starting state
        if len(prompt_words) >= 2:
            seed = (prompt_words[-2], prompt_words[-1])
            generated_words = prompt_words.copy()
            if seed not in self.chain:
                # If we have never seen the prompt bigram before, fall back to random
                seed = random.choice(list(self.chain.keys()))
                generated_words = list(seed)
        elif len(prompt_words) == 1:
            # Use the single word plus a random follower as seed
            first = prompt_words[-1]
            # Find all keys starting with the given word
            candidates = [key for key in self.chain.keys() if key[0] == first]
            if candidates:
                seed = random.choice(candidates)
                generated_words = [first, seed[1]]
            else:
                seed = random.choice(list(self.chain.keys()))
                generated_words = list(seed)
        else:
            # No prompt provided; pick a random starting bigram
            seed = random.choice(list(self.chain.keys()))
            generated_words = list(seed)

        # Generate words until reaching the desired length
        while len(generated_words) < max_length:
            key = (generated_words[-2], generated_words[-1])
            followers = self.chain.get(key)
            if not followers:
                # Dead end; pick a new random bigram to continue generation
                seed = random.choice(list(self.chain.keys()))
                generated_words.extend(list(seed))
                continue
            next_word = random.choice(followers)
            generated_words.append(next_word)

        # Capitalise the first word if the prompt started with a capital
        if prompt and prompt[0].isupper():
            generated_words[0] = generated_words[0].capitalize()
        return " ".join(generated_words)


class TextSummarizer:
    """Summarise text by selecting the most informative sentences.

    The algorithm uses a simple frequency‑based heuristic: it counts how
    often each word appears in the document (excluding common stopwords) and
    scores sentences by summing the frequencies of their constituent words.
    The top ``max_sentences`` sentences are returned in their original order.
    """

    def __init__(self) -> None:
        self.stopwords = self._default_stopwords()

    @staticmethod
    def _default_stopwords() -> set[str]:
        """Return a set of common English stopwords.

        The list here is a minimal set derived from typical stopword lists.
        It is defined inline to avoid external dependencies.
        """
        return {
            "a",
            "about",
            "above",
            "after",
            "again",
            "against",
            "all",
            "am",
            "an",
            "and",
            "any",
            "are",
            "as",
            "at",
            "be",
            "because",
            "been",
            "before",
            "being",
            "below",
            "between",
            "both",
            "but",
            "by",
            "could",
            "did",
            "do",
            "does",
            "doing",
            "down",
            "during",
            "each",
            "few",
            "for",
            "from",
            "further",
            "had",
            "has",
            "have",
            "having",
            "he",
            "her",
            "here",
            "hers",
            "herself",
            "him",
            "himself",
            "his",
            "how",
            "i",
            "if",
            "in",
            "into",
            "is",
            "it",
            "its",
            "itself",
            "just",
            "me",
            "more",
            "most",
            "my",
            "myself",
            "no",
            "nor",
            "not",
            "now",
            "of",
            "off",
            "on",
            "once",
            "only",
            "or",
            "other",
            "our",
            "ours",
            "ourselves",
            "out",
            "over",
            "own",
            "same",
            "she",
            "should",
            "so",
            "some",
            "such",
            "than",
            "that",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "there",
            "these",
            "they",
            "this",
            "those",
            "through",
            "to",
            "too",
            "under",
            "until",
            "up",
            "very",
            "was",
            "we",
            "were",
            "what",
            "when",
            "where",
            "which",
            "while",
            "who",
            "whom",
            "why",
            "will",
            "with",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
        }

    @staticmethod
    def _tokenize_sentences(text: str) -> List[str]:
        """Split text into sentences using punctuation boundaries."""
        # Use a simple regular expression to split on sentence terminators followed by
        # whitespace.  This works well for plain English text.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s]

    @staticmethod
    def _tokenize_words(sentence: str) -> List[str]:
        """Split a sentence into alphanumeric words."""
        return re.findall(r"\b\w+\b", sentence.lower())

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Return a summary consisting of the top sentences by importance.

        Parameters
        ----------
        text: str
            Source text containing one or more sentences.
        max_sentences: int, optional
            Maximum number of sentences to return.  If the input has fewer
            sentences than ``max_sentences`` then the entire text is returned.

        Returns
        -------
        str
            A summary constructed from the selected sentences.
        """
        sentences = self._tokenize_sentences(text)
        if len(sentences) <= max_sentences:
            return text
        # Compute word frequencies excluding stopwords
        freq = Counter()
        for sentence in sentences:
            for word in self._tokenize_words(sentence):
                if word not in self.stopwords:
                    freq[word] += 1
        # Score sentences by summing frequencies
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            score = 0
            for word in self._tokenize_words(sentence):
                if word not in self.stopwords:
                    score += freq.get(word, 0)
            scored_sentences.append((score, idx, sentence))
        # Select top sentences by score, keeping original order via index
        top = sorted(scored_sentences, key=lambda x: (-x[0], x[1]))[:max_sentences]
        top_sorted = sorted(top, key=lambda x: x[1])
        summary_sentences = [sentence for _, _, sentence in top_sorted]
        return " ".join(summary_sentences)


# Instantiate global services.  These objects will be reused across requests,
# which is more efficient than constructing a new Markov chain or computing
# stopwords on each call.
text_generator = MarkovChainGenerator()
text_summarizer = TextSummarizer()
