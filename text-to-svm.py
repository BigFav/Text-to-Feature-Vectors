#!/usr/bin/env python
from __future__ import unicode_literals
import argparse
import re
import sys
from collections import Counter, defaultdict
from operator import itemgetter


"""
Description:    Converts text files into SVM readable form.
Author:         Favian Contreras <fnc4@cornell.edu>
"""
# General cross-version compatibility funs, and regexes
eplisis = re.compile("\.\.+")
multiple_exc = re.compile("\!\!+")
multiple_ques = re.compile("\?\?+")
interrobang = re.compile("(\!+\?|\?+\!)[?!]*")
single_ques_or_exc = re.compile("(?<![?!\s])([?!])")

is_python2 = sys.version_info < (3,)
if is_python2:
    range = xrange
get_next = lambda i: i.next() if is_python2 else next(i)
get_keys = lambda d: d.iterkeys() if is_python2 else d.keys()
get_items = lambda d: d.iteritems() if is_python2 else d.items()
open_r_file = lambda f: (open(f, 'r') if is_python2 else
                         open(f, 'r', errors="replace"))
"""
Regexes from NLTK's TreebankWordTokenizer. The method word_tokenizer was
extremely slow, so I decided to extract it, and it performed MUCH better.

Author:         Edward Loper <edloper@gradient.cis.upenn.edu>
See more here:  https://code.google.com/p/nltk/source/browse/trunk/nltk/nltk/
                        tokenize/treebank.py
"""
contraction2_a = re.compile(r"(?i)(.)('ll|'re|'ve|n't|'s|'m|'d)\b")
contraction2_b = re.compile(r"(?i)\b(can)(not)\b")
contraction2_c = re.compile(r"(?i)\b(D)('ye)\b")
contraction2_d = re.compile(r"(?i)\b(Gim)(me)\b")
contraction2_e = re.compile(r"(?i)\b(Gon)(na)\b")
contraction2_f = re.compile(r"(?i)\b(Got)(ta)\b")
contraction2_g = re.compile(r"(?i)\b(Lem)(me)\b")
contraction2_h = re.compile(r"(?i)\b(Mor)('n)\b")
contraction2_i = re.compile(r"(?i)\b(T)(is)\b")
contraction2_j = re.compile(r"(?i)\b(T)(was)\b")
contraction2_k = re.compile(r"(?i)\b(Wan)(na)\b")
contraction3_a = re.compile(r"(?i)\b(Whad)(dd)(ya)\b")
contraction3_b = re.compile(r"(?i)\b(Wha)(t)(cha)\b")
separate_punct = re.compile(r"([^\w\'\-\/,&?!])")
seperate_commas = re.compile(r"(,\s)")
single_quotes = re.compile(r"('\s)")


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


def argument_checker():
    class ArgumentChecker(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) > 3:
                parser.error("argument -%s: too many arguments: requires "
                             "between 1 and 3 arguments." % self.dest)

            import os.path
            if not os.path.isfile(values[0]):
                parser.error("argument -%s: [Errno 2] No such file: '%s'" %
                             (self.dest, values[0]))

            has_output_file = False
            if len(values) > 1:
                if values[1].isdigit():
                    values[1] = int(values[1])
                else:
                    has_output_file = True
            if len(values) == 3:
                if has_output_file:
                    parser.error("argument -%s: invalid int value: '%s'" %
                                 (self.dest, values[1]))
            setattr(args, self.dest, tuple(values))
    return ArgumentChecker


def parse_args():
    parser = argparse.ArgumentParser(add_help=False,
                                     epilog=("CATEGORY_NUM refers to one of "
                                             "the potentially multiple "
                                             "classification categories that "
                                             "a line of text may have (e.g. "
                                             "is spam, is male, etc.). It is "
                                             "1 by default. The default "
                                             "output_file is svm_train.train "
                                             "for a given train_set, the "
                                             "others follow this pattern."))
    parser.add_argument("-h", "-?", "--help", action="store_true",
                        help="Show this help message and exit.")
    parser.add_argument("-m", "--multi", action="store_true",
                        help="Creates multi-class featrue vectors.")
    parser.add_argument("-lang", default="english", action="store",
                        metavar="LANGUAGE", help=("Language of text (English "
                                                  "by default)."))
    parser.add_argument("-stop", "--stopwords", action="store_true",
                        help="Remove stopwords (uses NLTK's stopwords).")

    reduce_forms = parser.add_mutually_exclusive_group()
    reduce_forms.add_argument("-stem", action="store_true",
                              help="Stem words using NLTK's SnowballStemmer.")
    reduce_forms.add_argument("-lemma", "--lemmatize", action="store_true",
                              help=("Lemmatize words using NLTK's WordNet "
                                    "lemmatizer (only for English)."))
    parser.add_argument("-train_set", nargs='+', action=argument_checker())
    parser.add_argument("-val_set", nargs='+', action=argument_checker())
    parser.add_argument("-test_set", nargs='+', action=argument_checker())

    parser.usage = ("text-to-svm.py [-h] [-m] [-lang LANGUAGE] [-stop] "
                    "[-lemma | -stem]\n\t\t      -train_set input_file "
                    "[CATEGORY_NUM] [output_file]\n\t\t      [-val_set "
                    "input_file [CATEGORY_NUM] [output_file]]\n\t\t      "
                    "-test_set input_file [CATEGORY_NUM] [output_file]")

    opts = parser.parse_args()
    if opts.help:
        help = parser.format_help().split("\n\n")
        usage, help, epilog = help
        help = help.split('\n')
        help[-3] = "  -train_set input_file [CATEGORY_NUM] [output_file]"
        help[-2] = "  -val_set input_file [CATEGORY_NUM] [output_file]"
        help[-1] = "  -test_set input_file [CATEGORY_NUM] [output_file]"
        help.append('\n')
        sys.stdout.write(usage + '\n' + '\n'.join(help) + epilog)
        sys.exit(0)
    if not (opts.train_set and opts.test_set):
        parser.error("Must give at least a train set and test set (should "
                     "give the validation set, if one is available).")

    opts.lang = opts.lang.lower()
    if opts.stopwords:
        from nltk.corpus import stopwords
        opts.stopwords = frozenset(stopwords.words(opts.lang))
    if opts.stem:
        from nltk.stem import SnowballStemmer
        opts.stem = SnowballStemmer(opts.lang).stem
    elif opts.lemmatize:
        if opts.lang != "english":
            parser.error("argument -lang: invalid str value: WordNet "
                         "lemmatizer only works for the English langauge.")
        from nltk.stem.wordnet import WordNetLemmatizer
        opts.lemmatize = WordNetLemmatizer().lemmatize
    return opts, parser


def parse_and_tokenize(line, category_num, num_categories,
                       stopwords, stem_or_lemma, lem_stem_memo):
    if is_python2:
        line = unicode(line, errors="replace")

    # Find the label, and text portions of the line
    begin_comma = end_comma = -1
    for _ in range(category_num):
        begin_comma = end_comma
        end_comma = line.find(',', end_comma+1)
        if end_comma == -1:
            sys.exit("Error: More categories are defined than are present.")
    last_comma = end_comma
    for _ in range(num_categories-category_num):
        last_comma = line.find(',', last_comma+1)
        if last_comma == -1:
            sys.exit("Error: More categories are defined than are present.")
    label = line[begin_comma+1:end_comma].strip()
    line = line[last_comma+1:]

    line = line.replace(":(", " \u2639 ")
    line = line.replace(":-(", " \u2639 ")
    line = line.replace(":)", " \u263A ")
    line = line.replace(":-)", " \u263A ")
    line = eplisis.sub(" \u2026 ", line)
    line = interrobang.sub(" \u203D ", line)
    line = multiple_exc.sub(" !! ", line)
    line = multiple_ques.sub(" ?? ", line)
    line = line.replace("\uFFFD", " \uFFFD ")
    line = single_ques_or_exc.sub(r" \1 ", line)

    line = contraction2_a.sub(r"\1 \2", line)
    line = contraction2_b.sub(r"\1 \2", line)
    line = contraction2_c.sub(r"\1 \2", line)
    line = contraction2_d.sub(r"\1 \2", line)
    line = contraction2_e.sub(r"\1 \2", line)
    line = contraction2_f.sub(r"\1 \2", line)
    line = contraction2_g.sub(r"\1 \2", line)
    line = contraction2_h.sub(r"\1 \2", line)
    line = contraction2_i.sub(r"\1 \2", line)
    line = contraction2_j.sub(r"\1 \2", line)
    line = contraction2_k.sub(r"\1 \2", line)
    line = contraction3_a.sub(r"\1 \2 \3", line)
    line = contraction3_b.sub(r"\1 \2 \3", line)
    line = separate_punct.sub(r" \1 ", line)
    line = seperate_commas.sub(r" \1", line)
    line = single_quotes.sub(r" \1", line).split()

    # Remove stopwords, or stem/lemmatize
    if stopwords:
        fst_ex = [word for word in line if word.lower() not in stopwords]

    if stem_or_lemma:
        for i, word in enumerate(line):
            if word[0].isalpha() or word[-1].isalpha():  # Minimize exp calls
                lemmatized = lem_stem_memo[word.lower()]  # Memoized stem/lemma
                if word.istitle():
                    line[i] = lemmatized.capitalize()
                elif word == word.upper():
                    line[i] = lemmatized.upper()
                else:
                    line[i] = lemmatized

    return line, label


def get_feature_vectors(total_words, data_words, data_labels, multi):
    ft_vecs = []
    for i, word_dict in enumerate(data_words):
        if not data_labels or (data_labels[i] == '1'):
            ft_vecs.append('1')
        elif multi:
            ft_vecs.append(data_labels[i])
        else:
            ft_vecs.append('-1')  # Just in case file uses '0' for neg

        ft_ids = []
        for word, count in get_items(word_dict):
            ft_id = total_words[word]
            ft_ids.append((ft_id, (" %s:%s" % (ft_id, count))))
        # Must be in-order, no fancy scapegoat trees w/ indirection
        ft_ids = sorted(ft_ids, key=itemgetter(0))

        ft_vecs.extend(tup[1] for tup in ft_ids)
        ft_vecs.append('\n')

    return ''.join(ft_vecs)


def main():
    # Create cross-version functions, and starter vars
    opt, parser = parse_args()
    stem = 0.0
    ft_id = 1
    total_words = {}
    default_output_filename = {
        opt.train_set: "svm_train.train",
        opt.val_set: "svm_val.val",
        opt.test_set: "svm_test.test"
    }
    stem_or_lemma = False
    if opt.lemmatize:
        stem_or_lemma = opt.lemmatize
    elif opt.stem:
        stem_or_lemma = opt.stem
    lem_stem_memo = keydefaultdict(stem_or_lemma)
    # Read files, and add counts of the words in the line as ft val
    for file_tup in filter(bool, [opt.train_set, opt.val_set, opt.test_set]):
        file_words = []
        file_labels = []
        category_num = 1
        output_file = None
        is_test = file_tup is opt.test_set
        # Overwrite default arg values to proper vars, if values are given
        if len(file_tup) == 2:
            if isinstance(file_tup[1], int):
                category_num = file_tup[1]
            else:
                output_file = file_tup[1]
        elif len(file_tup) == 3:
            category_num = file_tup[1]
            output_file = file_tup[2]
        if not output_file:
            output_file = default_output_filename[file_tup]

        with open_r_file(file_tup[0]) as text:
            test_flag = False
            lines = iter(text)
            num_categories = get_next(lines).count(',')
            if category_num > num_categories:
                parser.error("CATEGORY_NUM is greater than the inumber of "
                             "categories found at the top of the file.")

            # If test_set, extract 1st example to see if it is labelled
            if is_test:
                fst_ex, label = parse_and_tokenize(get_next(lines),
                                                   category_num,
                                                   num_categories,
                                                   opt.stopwords,
                                                   stem_or_lemma,
                                                   lem_stem_memo)
                file_words = [Counter(fst_ex)]
                if label != '?':
                    file_labels = [label]

            # Get the label, and count of each word in the example
            for line in lines:
                line, label = parse_and_tokenize(line,
                                                 category_num,
                                                 num_categories,
                                                 opt.stopwords,
                                                 stem_or_lemma,
                                                 lem_stem_memo)
                file_words.append(Counter(line))
                if not is_test or file_labels:
                    file_labels.append(label)

        # The order the word is seen in the dicts will be the ft id
        for dictionary in file_words:
            for word in get_keys(dictionary):
                if word not in total_words:  # To keep ft_ids continuous
                    total_words[word] = ft_id
                    ft_id += 1

        # Create the ft vecs and file
        ft_vecs = get_feature_vectors(total_words, file_words,
                                      file_labels, opt.multi)
        with open(output_file, 'w') as output:
            output.write(ft_vecs)

if __name__ == "__main__":
    main()
