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
    line = re.sub("(?<![?!])([?!])", r" \1 ", line)
    line = re.sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", line).split()

    line = filter(bool, line)
    line.insert(0, line.pop())
    return line

def main():
    train_words = []
    train_labels = []
    test_words = []
    # Read in the files, and add the counts of the words in the line as ft val
    with open(sys.argv[1], 'r') as reviews:
        for line in reviews.readlines()[1:]:
            line = unicode(line, errors='replace')
            train_labels.append(line[0])
            line = parse_and_tokenize(line[4:])
            train_words.append(Counter(line))

    with open(sys.argv[2], 'r') as reviews:
        for line in reviews.readlines()[1:]:
            line = unicode(line, errors='replace')
            line = parse_and_tokenize(line[4:])
            test_words.append(Counter(line))

    # The order the word is seen in the dicts will be the ft id
    total_words = {}
    i = 0
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
    ft_vecs = []
    for i, word_dict in enumerate(train_words):
        if train_labels[i] == '0':
            train_labels[i] = '-1'
        ft_vecs.append(train_labels[i])

        ft_ids = []
        for word, count in word_dict.iteritems():
            ft_id = total_words[word] + 1
            ft_ids.append((ft_id, (' %s:%s' % (ft_id, count))))
        ft_ids = sorted(ft_ids, key=itemgetter(0))  # Must be in order

        ft_vecs.extend(tup[1] for tup in ft_ids)
        ft_vecs.append('\n')

    ft_vecs = ''.join(ft_vecs)
    with open('svm_train.train', 'w') as train:
        train.write(ft_vecs)

    ft_vecs = []
    for word_dict in test_words:
        ft_vecs.append('-1')

        ft_ids = []
        for word, count in word_dict.iteritems():
            ft_id = total_words[word] + 1
            ft_ids.append((ft_id, (' %s:%s' % (ft_id, count))))
        ft_ids = sorted(ft_ids, key=itemgetter(0))

        ft_vecs.extend(tup[1] for tup in ft_ids)
        ft_vecs.append('\n')

    ft_vecs = ''.join(ft_vecs)
    with open('svm_test.test', 'w') as test:
       test.write(ft_vecs)
 
if __name__ == '__main__':
    main()
