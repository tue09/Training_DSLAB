from nltk.stem.porter import PorterStemmer

# Tạo một instance của PorterStemmer.
stemmer = PorterStemmer()

# Từ cần stem.
word = "gone"

# Sử dụng phương thức stem().
stemmed_word = stemmer.stem(word)

print(stemmed_word)  # In ra: "run"
