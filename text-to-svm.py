import re
from collections import Counter
from operator import itemgetter


total_words = []
train_words = []
train_labels = []
test_words = []

start_tokens = " </s> <s>"

with open('reviews.train', 'r') as reviews:
    for i, tokens in enumerate(reviews.readlines()[1:]):
        tokens = unicode(tokens, errors='replace')
        train_labels.append(tokens[0])
        tokens = tokens[4:]
        tokens = tokens.replace(":)", " \u1F601 ")
        tokens = tokens.replace(":", " : ")
        tokens = tokens.replace(")", " ) ")
        tokens = tokens.replace("(", " ( ")
        tokens = tokens.replace(";", " ; ")
        tokens = tokens.replace("\"", " \" ")
        tokens = re.sub("\.\.+", u" \u2026 ", tokens) #elipsis
        tokens = tokens.replace(".", " ." + start_tokens)
        tokens = re.sub("(\!+\?|\?+\!)[?!]*", " \u203D" + start_tokens, tokens)
        tokens = re.sub("\!\!+"," !!" + start_tokens, tokens)
        tokens = re.sub("\?\?+"," ??" + start_tokens, tokens)
        tokens = re.sub("(?<![?!])([?!])", r" \1" + start_tokens, tokens)
        tokens = re.sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", tokens).split()

        tokens = filter(bool, tokens)
        tokens.insert(0, tokens.pop())

        train_words.append(Counter(tokens))

with open('reviews.test', 'r') as reviews:
    for i, line in enumerate(reviews.readlines()[1:]):
        line = unicode(line, errors='replace')
        line = line[4:]
        line = line.replace(":)", " \u1F601 ")
        line = line.replace(")", " ) ")
        line = line.replace("(", " ( ")
        line = line.replace(":", " : ")
        line = line.replace(";", " ; ")
        line = line.replace("\"", " \" ")
        line = re.sub("\.\.+", u" \u2026 ", line) #elipsis
        line = line.replace(".", " ." + start_tokens)
        line = re.sub("(\!+\?|\?+\!)[?!]*", " \u203D" + start_tokens, line)
        line = re.sub("\!\!+"," !!" + start_tokens, line)
        line = re.sub("\?\?+"," ??" + start_tokens, line)
        line = re.sub("(?<![?!])([?!])", r" \1" + start_tokens, line)
        line = re.sub("(?<=[a-zI])('[a-z][a-z]?)\s", r" \1 ", line).split()

        line = filter(bool, line)
        line.insert(0, line.pop())

        test_words.append(Counter(line))

for lst in train_words:
	for word in lst.iterkeys():
		total_words.append(word)

for lst in test_words:
	for word in lst.iterkeys():
		total_words.append(word)


# Use the index of the word in total_words as value for feature
with open('svm_train.train', 'w') as train:
	for i,word_dict in enumerate(train_words):
		if train_labels[i] == '0':
			train_labels[i] = '-1'

		train.write(train_labels[i])

		ft_lst = []

		for word,count in word_dict.iteritems():
			ft_id = total_words.index(word) + 1
			ft_lst.append((ft_id, (' %s:%s' % (ft_id, count))))

		ft_lst = sorted(ft_lst, key=itemgetter(0))

		for item in ft_lst:
			train.write(item[1])

		train.write('\n')


with open('svm_test.test', 'w') as test:
	for word_dict in test_words:
		test.write('-1')

		ft_lst = []

		for word,count in word_dict.iteritems():
			ft_id = total_words.index(word) + 1
			ft_lst.append((ft_id, (' %s:%s' % (ft_id, count))))

		ft_lst = sorted(ft_lst, key=itemgetter(0))

		for item in ft_lst:
			test.write(item[1])

		test.write('\n') 
