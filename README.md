# Text-to-SVM

Convert your text sets into SVM readable files. Given a file in the form:

    isSpam, positive ,text
    1,0,Ugh, blah blah blah.....

This program will output a series of feature vectors for a given category. The
term category is used to describe the different classification jobs (e.g. is
spam, is positive, etc.); ``CATEGORY_NUM`` for spam is 1 in the example above.
This program assumes binary classification (converting 0s to -1s), however it
can convert for multi-class data sets as well by using the ``-multi`` option.
Runs on Python 2.7+.

## Usage

The usage for this program is:

    text-to-svm.py [-h] [-multi] [-train_set input_file [CATEGORY_NUM] [output_file]]
             [-val_set input_file [CATEGORY_NUM] [output_file]]
             [-test_set input_file [CATEGORY_NUM] [output_file]]

Running the program with any of the help options (``-?``, ``-h``, ``--help``)
will provide a brief help message.
