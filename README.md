# Text-to-SVM

Want to use SVM to classify your text? Convert your text sets into SVM readable
files. Given a file in the form:

    isSpam, positive ,text
    1,0,Ugh, blah blah blah.....

This program will output a series of feature vectors for a given category. I
recommend using <a href="http://svmlight.joachims.org/">SVM-Light</a> for
binary classification, and
<a href="http://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html">SVM-Multiclass</a>
for multi-class classifications with the files this program outputs.

The term category is used to describe the different classification jobs (e.g.
is spam, is positive, etc.); ``CATEGORY_NUM`` for spam is 1 in the example
above. This program assumes binary classification (converting 0s to -1s),
however it can convert for multi-class data sets as well by using the
``-m`` option. Runs on Python 2.7+.

## Usage

The usage for this program is:

    text-to-svm.py [-h] [-m] [-lang LANGUAGE] [-stop] [-lemma | -stem]
                   [-train_set input_file [CATEGORY_NUM] [output_file]]
                   [-val_set input_file [CATEGORY_NUM] [output_file]]
                   [-test_set input_file [CATEGORY_NUM] [output_file]]

Running the program with any of the help options (``-?``, ``-h``, ``--help``)
will provide a brief help message.

## NLTK Tools

While the basic functionality requires no outside software installation, this
program uses NLTK tools to remove stopwords, stem, and lemmatize. These tools
work across languages as well which can be input via the ``-lang`` option
(with the exception of the lemmatizer, which only works for English).

To see more:
* Stopwords - <a href="http://www.nltk.org/book/ch02.html">http://www.nltk.org/book/ch02.html</a>
* Snowball Stemmer - <a href="http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball">http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball</a>
* WordNet Lemmatizer - <a href="http://www.nltk.org/_modules/nltk/stem/wordnet.html">http://www.nltk.org/_modules/nltk/stem/wordnet.html</a>
