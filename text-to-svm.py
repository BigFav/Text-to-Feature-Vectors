from __future__ import unicode_literals
import re
import sys
from collections import Counter
from operator import itemgetter
from string import punctuation


punctuation = (punctuation.replace("?", "").replace("'", "").
                           replace("!", "").replace(".", ""))


def parse_and_tokenize(line):
    line = line.replace(":)", " \u1F601 ")
    for ch in punctuation:
        line = line.replace(ch, ' ' + ch + ' ')
    line = re.sub("\.\.+", " \u2026 ", line)
    line = line.replace(".", " . ")
    line = re.sub("(\!+\?|\?+\!)[?!]*", " \u203D ", line)
    line = re.sub("\!\!+"," !! ", line)
    line = re.sub("\?\?+"," ?? ", line)
    line = re.sub("(?<![?!\s])([?!])", r" \1 ", line)
    return re.sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", line).split()


def get_feature_vectors(total_words, data_words, data_labels=None):
    ft_vecs = []
    for i, word_dict in enumerate(data_words):
        if not data_labels or (data_labels[i] == '0'):
            ft_vecs.append('-1')
        else:
            ft_vecs.append(data_labels[i])

        ft_ids = []
        for word, count in word_dict.iteritems():
            ft_id = total_words[word]
            ft_ids.append((ft_id, (' %s:%s' % (ft_id, count))))
        ft_ids = sorted(ft_ids, key=itemgetter(0))  # Must be in order

        ft_vecs.extend(tup[1] for tup in ft_ids)
        ft_vecs.append('\n')

    return ''.join(ft_vecs)


def main():
    train_words = []
    train_labels = []
    test_words = []
    # Read in the files, and add the counts of the words in the line as ft val
    with open(sys.argv[1], 'r') as reviews:
        for line in reviews.readlines()[1:]:
            line = unicode(line, errors='replace')
            # Assumes only one prediction category (e.g. spam, male, etc.)
            comma = line.find(',')
            label = line[:comma].strip()
            train_labels.append(label)
            line = parse_and_tokenize(line[comma+1:])
            train_words.append(Counter(line))

    with open(sys.argv[2], 'r') as reviews:
        for line in reviews.readlines()[1:]:
            line = unicode(line, errors='replace')
            line = parse_and_tokenize(line[line.find(',')+1:])
            test_words.append(Counter(line))

    # The order the word is seen in the dicts will be the ft id
    total_words = {}
    i = 1
    for lst in train_words:
        for j, word in enumerate(lst.iterkeys()):
            if word not in total_words:
                total_words[word] = i + j
        i += j + 1

    for lst in test_words:
        for j, word in enumerate(lst.iterkeys()):
            if word not in total_words:
                total_words[word] = i + j
        i += j + 1

    # Create the ft vecs and files
    ft_vecs = get_feature_vectors(total_words, train_words, train_labels)
    with open('svm_train.train', 'w') as train:
        train.write(ft_vecs)

    ft_vecs = get_feature_vectors(total_words, test_words)
    with open('svm_test.test', 'w') as test:
       test.write(ft_vecs)
 
if __name__ == '__main__':
    main()
