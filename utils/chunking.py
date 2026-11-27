def chunk_text(text, max_word=100):
    words = text.split()
    return [" ".join(words[i:i+max_word]) for i in range(0, len(words), max_word)]

