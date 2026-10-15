"""Text cleaning and tokenising, shared by every retrieval method."""
import re

from .paths import DATA_DIR

TOKEN_PATTERN = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b")


def normalize_text(text):
    """Turn any whitespace (tabs, new lines, double spaces) into single spaces."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_stopwords(filename="scientific_stopwords.txt"):
    """Read the stopword file into a set (a set makes 'is this word in it?' very fast)."""
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


STOPWORDS = load_stopwords()


def tokenize(text, stopwords=STOPWORDS):
    """Lowercase the text, split it into words, drop stopwords and 1-letter tokens."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def make_tokenizer(stopword_list="project", stem=False):
    """Build a tokenizer with a chosen recipe, so recipes can be compared fairly.

    stopword_list : "project" -> our own data/scientific_stopwords.txt
                    "english" -> the full standard NLTK English list
    stem          : True -> cut words to their root with Porter's stemmer
                    (the paper does this: "variants" and "variant" become the same word)
    """
    if stopword_list == "project":
        stopwords = STOPWORDS
    elif stopword_list == "english":
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords
        nltk.download("stopwords", quiet=True)
        stopwords = set(nltk_stopwords.words("english"))
    else:
        raise ValueError(f"Unknown stopword list: {stopword_list}")

    if not stem:
        return lambda text: tokenize(text, stopwords)

    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    memory = {}   # remember stems we already worked out (much faster)

    def stem_word(word):
        if word not in memory:
            memory[word] = stemmer.stem(word)
        return memory[word]

    return lambda text: [stem_word(t) for t in tokenize(text, stopwords)]
