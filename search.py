"""make_lda_data. Compile corpus + dictionary from textfile across all submissions.
Usage:
  make_lda_data <textfile>
"""

import string, os, codecs, gensim, save

from docopt import docopt
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer


def search(text,word):
    words = text.split()
    sections = []
    n = len(words)
    for i in range(n):
        if words[i] == word:
            try:
                section = words[i-30:i+30]
            except:
                try:
                    section = words[0:i+30]
                except:
                    section = words[i-30:]
            sections.append(section)
    return sections


def texts_iter(filename,word):
    for f in sorted(os.listdir('.')):
        if f[0] != '.' and os.path.isdir(f):
            try:
                with codecs.open(f + "/" + filename, "r", "utf-8", "ignore") as a:
                    print "Submission by ", f
                    raw_text = a.read()
                    yield (f, search(raw_text,word))
            except IOError:
                print "No file for ", f
                pass

if __name__ == "__main__":
    args = docopt(__doc__)

    docs, raw_texts, texts = zip(*list(texts_iter(args['<textfile>'])))
    dictionary = gensim.corpora.dictionary.Dictionary(texts)
    dictionary.compactify()

    corpus = map(dictionary.doc2bow, texts)

    save.save(args['<datafile>'], docs=docs, texts=texts,
            raw_texts=raw_texts, corpus=corpus, dictionary=dictionary)





