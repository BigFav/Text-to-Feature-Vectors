#!/usr/bin/env python
from __future__ import unicode_literals
import argparse
import re
import sys
from collections import Counter
from operator import itemgetter
from string import punctuation


eplisis = re.compile("\.\.+")
multi_exc = re.compile("\!\!+")
multi_quest = re.compile("\?\?+")
interrobang = re.compile("(\!+\?|\?+\!)[?!]*")
single_exc_quest = re.compile("(?<![?!\s])([?!])")
contractions = re.compile("(?<=[a-zI])('[a-z][a-z]?)\s")
punctuation = (punctuation.replace("?", "").replace("'", "").
                           replace("!", "").replace(".", ""))


def argument_checker():
    class ArgumentChecker(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) > 3:
                parser.error("argument -%s: too many arguments: requires "
                             "between 1 and 3 arguments." % self.dest)

            import os.path
            if not os.path.isfile(values[0]):
                parser.error("argument -%s: [Errno 2] No such file or "
                             "directory: '%s'" % (self.dest, values[0]))

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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", "-?", action='store_true')
    parser.add_argument("-train_set", nargs='+', action=argument_checker())
    parser.add_argument("-val_set", nargs='+', action=argument_checker())
    parser.add_argument("-test_set", nargs='+', action=argument_checker())
    parser.usage = ("text-to-svm.py [-h] [-train_set input_file [CATEGORY_NUM]"
                    " [output_file]]\n\t\t[-val_set input_file [CATEGORY_NUM] "
                    "[output_file]]\n\t\t[-test_set input_file [CATEGORY_NUM] "
                    "[output_file]]\n\nCATEGORY_NUM refers to one of the "
                    "potentially multiple classification categories that a "
                    "line of text may have (e.g. is spam, is male, etc.). It "
                    "is 1 by default. The default output_file is "
                    "svm_train.train for an inputted train_set, the others "
                    "follow this pattern.")

    opts = parser.parse_args()
    num_args = len(sys.argv) - 2 if opts.help else len(sys.argv) - 1
    if not num_args:
        if opts.help:
            parser.print_usage()
            sys.exit(0)
        parser.error("No arguments given.")
    return opts, parser


def parse_and_tokenize(line):
    line = line.replace("\uFFFD", " \uFFFD ")
    line = line.replace(":)", " \u1F601 ")
    for ch in punctuation:
        line = line.replace(ch, ' ' + ch + ' ')
    line = eplisis.sub(" \u2026 ", line)
    line = line.replace(".", " . ")
    line = interrobang.sub(" \u203D ", line)
    line = multi_exc.sub(" !! ", line)
    line = multi_quest.sub(" ?? ", line)
    line = single_exc_quest.sub(r" \1 ", line)
    return contractions.sub(r" \1 ", line).split()


def get_feature_vectors(total_words, data_words, data_labels):
    ft_vecs = []
    get_items = lambda d: (d.iteritems() if sys.version_info < (3,) else
                           d.items())
    for i, word_dict in enumerate(data_words):
        if not data_labels or (data_labels[i] == '1'):
            ft_vecs.append('1')
        else:
            ft_vecs.append('-1')  # Just in case file uses '0' for neg

        ft_ids = []
        for word, count in get_items(word_dict):
            ft_id = total_words[word]
            ft_ids.append((ft_id, (' %s:%s' % (ft_id, count))))
        ft_ids = sorted(ft_ids, key=itemgetter(0))  # Must be in order

        ft_vecs.extend(tup[1] for tup in ft_ids)
        ft_vecs.append('\n')

    return ''.join(ft_vecs)


def main():
    # Create cross-version functions, and starter vars
    is_python2 = sys.version_info < (3,)
    get_next = lambda i: i.next() if is_python2 else next(i)
    get_keys = lambda d: d.iterkeys() if is_python2 else d.keys()
    open_r_file = lambda f: (open(f, 'r') if is_python2 else
                             open(f, 'r', errors="replace"))

    opt, parser = parse_args()
    ft_id = 1
    total_words = {}
    default_output_filename = {
        opt.train_set: "svm_train.train",
        opt.val_set: "svm_val.val",
        opt.test_set: "svm_test.test"
    }
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
                parser.error("CATEGORY_NUM is greater than the number of "
                             "categories found at the top of the file.")
            last_category_slot = 2 * num_categories
            label_slot = 2 * (category_num - 1)

            # If test_set, extract 1st example to see if it is labelled
            if is_test:
                first_example = parse_and_tokenize(get_next(lines))
                file_words = [Counter(first_example[last_category_slot:])]
                if first_example[label_slot] == '?':
                    label = None
                else:
                    file_labels = [label]

            # Get the label, and count of each word in the example
            for line in lines:
                if is_python2:
                    line = unicode(line, errors='replace')
                line = parse_and_tokenize(line)
                file_words.append(Counter(line[last_category_slot:]))
                if not is_test or file_labels:
                    file_labels.append(line[label_slot])

        # The order the word is seen in the dicts will be the ft id
        for dictionary in file_words:
            for word in get_keys(dictionary):
                if word not in total_words:  # To keep ft_ids continuous
                    total_words[word] = ft_id
                    ft_id += 1

        # Create the ft vecs and file
        ft_vecs = get_feature_vectors(total_words, file_words, file_labels)
        with open(output_file, 'w') as output:
            output.write(ft_vecs)

if __name__ == '__main__':
    main()
