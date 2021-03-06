# Feature-Vectorize-Text

Want to use SVM to classify your text? Want kNN? Convert your text sets into
feature vectors, so that you can use machine learning algorithms to classify.
Given a file in the form:

    isSpam, positive ,text
    1,0,Ugh, blah blah blah

This program will output a series of feature vectors for a given category,
where the order of occurence of the word is the feature id, and the frequency
of the word in the example is the value. The ``-p`` option allows using word
pair occurences and frequencies instead of individual words.

The term category is used to describe the different classification jobs (e.g.
is spam, is positive, etc.); ``CATEGORY_NUM`` for spam is 1 in the example
above. This program assumes binary classification (converting 0s to -1s),
however it can convert for multi-class data sets as well by using the
``-m`` option. Runs on Python 2.7+.

## Usage

The usage for this program is:

    text-to-svm.py [-h] [-p] [-m] [-lang LANGUAGE] [-stop] [-lemma | -stem]
                   -train_set input_file [CATEGORY_NUM] [output_file]
                   [-val_set input_file [CATEGORY_NUM] [output_file]]
                   -test_set input_file [CATEGORY_NUM] [output_file]

Running the program with any of the help options (``-?``, ``-h``, ``--help``)
will provide a brief help message.

An example run is:

    ./text-to-svm.py -stop -lemma -train_set train.train output.train -test_set test.test

## NLTK Tools

While the basic functionality requires no outside software installation, this
program uses NLTK tools to remove stopwords, stem, and lemmatize. These tools
work across languages as well which can be input via the ``-lang`` option
(with the exception of the lemmatizer, which only works for English).

To see more:
* Stopwords - <a href="http://www.nltk.org/book/ch02.html">http://www.nltk.org/book/ch02.html</a>
* Snowball Stemmer - <a href="http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball">http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball</a>
* WordNet Lemmatizer - <a href="http://www.nltk.org/_modules/nltk/stem/wordnet.html">http://www.nltk.org/_modules/nltk/stem/wordnet.html</a>
