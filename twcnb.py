# -*- coding: utf-8 -*-
import math
from collections import defaultdict
from itertools import permutations, chain


class TWCNB(object):
    """
    twcnb = TWCNB()
    training_data = [(1, {'a': 2, 'b': 2}), (2, {'a': 1, 'c': 4})]
    twcnb.train(training_data)
    testing_data = {'a': 1, 'b': 2}
    result = twcnb.train(testing_data)
    """

    def __init__(self):
        """
        """
        self.all_categories = set()
        self.normalized_category_word_weight = defaultdict(lambda: defaultdict(int))

    def train(self, data):
        """dataを用いて分類器を学習する
        """
        category_word_count = self._calc_category_word_count(data)
        all_words = set(chain.from_iterable(word_count.iterkeys() for word_count in category_word_count.itervalues()))
        complement_category_word_count = self._calc_complement_category_word_count(category_word_count, all_words)
        category_word_weight = self._calc_category_word_weight(complement_category_word_count, len(all_words))
        normalized_category_word_weight = self._calc_normalized_category_word_weight(category_word_weight)
        self.normalized_cat_word_weight = normalized_category_word_weight

    def classify(self, document):
        """documentが分類されるcategoryを返す
        """
        scores = {category: self._calc_score(document, category)
                  for category in self.all_categories}
        best = min(scores, key=scores.get)
        return best

    def _calc_score(self, document, category):
        """documentがcategoryに属するスコアを算出する
        """
        score = 0
        for word, count in document.iteritems():
            score += count * self.normalized_cat_word_weight[category][word]
        return score

    def _calc_idf(self, documents):
        """idf値を算出する
        """
        df = defaultdict(int)
        for document in documents:
            for word in document.iterkeys():
                df[word] += 1
        idf = defaultdict(int)
        document_count = len(documents)
        for word, count in df.iteritems():
            idf[word] = math.log(1.0 * document_count / count)
        return idf

    def _calc_category_word_count(self, data):
        """各カテゴリでの各単語の出現頻度を算出する
        単語の出現頻度にはTF Transform, IDF Transform, Length Normalizationを行う
        """
        documents = [d[1] for d in data]
        idf = self._calc_idf(documents)
        category_word_count = defaultdict(lambda: defaultdict(int))
        for category, document in data:
            tfidf_count = defaultdict(float)
            for word, count in document.iteritems():
                self.all_categories.add(category)
                count = math.log(count + 1)  # 3. TF Transform
                count *= idf[word]  # 4. IDF Transform
                tfidf_count[word] = count
            document_length = math.sqrt(sum(math.pow(v, 2) for v in tfidf_count.itervalues()))
            for word, count in tfidf_count.iteritems():
                count /= document_length  # 5. Length Normalization
                category_word_count[category][word] += count
        return category_word_count

    def _calc_complement_category_word_count(self, category_word_count, all_words):
        """あるカテゴリ以外(Comlement)のカテゴリでの各単語の出現頻度を算出する
        """
        complement_category_word_count = defaultdict(lambda: defaultdict(int))  # 1. Complement
        for word in all_words:
            for category, other_category in permutations(self.all_categories, 2):
                complement_category_word_count[category][word] += category_word_count[other_category][word]
        return complement_category_word_count

    def _calc_category_word_weight(self, complement_category_word_count, alpha):
        """各カテゴリでの各単語の重みを算出する
        """
        category_word_weight = defaultdict(lambda: defaultdict(int))
        for category, word_count in complement_category_word_count.iteritems():
            theta_denominator = sum(word_count.itervalues()) + alpha
            for word, count in word_count.iteritems():
                theta = (count + 1) / theta_denominator
                category_word_weight[category][word] = math.log(theta)
        return category_word_weight

    def _calc_normalized_category_word_weight(self, category_word_weight):
        """各カテゴリでの各単語の重みを正規化する(Weight Normalization)
        """
        normalized_category_word_weight = defaultdict(lambda: defaultdict(int))
        for category, word_weight in category_word_weight.iteritems():
            weight_sum = sum(map(abs, word_weight.itervalues()))
            for word, weight in word_weight.iteritems():
                normalized_category_word_weight[category][word] = weight / weight_sum  # 2. Weight Normalization
        return normalized_category_word_weight
