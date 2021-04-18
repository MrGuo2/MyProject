import os
import re
import types

import jieba

from .log import logger
from .file import enumerate_file


class Tokenizer(object):
    """Tokenizer class."""
    def __init__(self, temp_dir, jieba_dict_file, remove_stopwords, stopwords_file, ivr):
        self.ivr = ivr
        # Init jieba config.
        self.__init_jieba(temp_dir, jieba_dict_file)
        # Init stopwords.
        if remove_stopwords:
            self.load_stopword_file(stopwords_file)

    def __init_jieba(self, temp_dir, jieba_dict_file):
        """Init jieba config."""
        jieba.dt.tmp_dir = temp_dir
        if os.path.exists(jieba_dict_file):
            jieba.load_userdict(jieba_dict_file)
        if self.ivr:
            # Alter jieba.cut method.
            jieba.re_han_default = re.compile("([\u4E00-\u9FD50-9+#&\._%]+|[a-zA-Z]+)", re.U)
            jieba.re_skip_default = re.compile("\r\n|\s", re.U)
            jieba.re_eng_word = re.compile('^[A-Za-z]+$')
            jieba.dt.cut = types.MethodType(cut, jieba.dt)
            jieba.cut = jieba.dt.cut

    def load_stopword_file(self, stopwords_file):
        """Load stopwords file into self.stopwords.

        Args:
            stopwords_file: str, stopword file path.

        Returns:
            None
        """
        self.stopwords = set()
        assert os.path.exists(stopwords_file), f'Stopwords file {stopwords_file} does not exist.'
        for _, word in enumerate_file(stopwords_file):
            self.stopwords.add(word.rstrip('\n'))
        logger.info(f'Stopwords are loaded, length: {len(self.stopwords)}')

    def tokenizer(self, sentence, seg_type='word', ngram=1, remove_stopwords=False):
        """Tokenize the sentence into token list.

        Args:
            sentence: str, input sentence.
            seg_type: str, segmentation type, including "word", "char". Default: "word".
            ngram: int, max gram number. Default: 1.
            remove_stopwords: bool, whether the sentence needs to remove stopwords. Default: False.

        Returns:
            Token list.
        """
        def remove_space(matched):
            word = matched.group()
            return re.sub('\s', '', word)

        if self.ivr:
            # Replace "A P P" with "APP".
            sentence = re.sub('^(([a-zA-Z] )+[a-zA-Z])$|^(([a-zA-Z] )+[a-zA-Z])(?=[^a-zA-Z])|'
                              '(?<=[^a-zA-Z^])(([a-zA-Z] )+[a-zA-Z])$'
                              '|((?<=[^a-zA-Z^])(([a-zA-Z] )+[a-zA-Z])(?=[^a-zA-Z]))',
                              remove_space, sentence)
            # Replace digits with "NUM".
            sentence = re.sub(r'\d+([.]{1}[0-9]+)?%?', 'NUM', sentence.lower())

        if seg_type == 'word':
            tokens = self.__wordseg(sentence, remove_stopwords=remove_stopwords)
        elif seg_type == 'char':
            tokens = self.__charseg(sentence, remove_stopwords=remove_stopwords)
        else:
            raise NotImplementedError(f'The seg_type {seg_type} is not in ["char", "word"].')

        return self.__ngrams(tokens, ngram)

    def __wordseg(self, query, remove_stopwords=False):
        """Tokenize the sentence into word list by Jieba module.

        Args:
            query: str, input sentence.
            remove_stopwords: bool, whether the sentence needs to remove stopwords. Default: False.

        Returns:
            seg_list: Word list.
        """
        seg_list = ' '.join(jieba.cut(query)).split()
        if remove_stopwords:
            seg_list = [x for x in seg_list if x not in self.stopwords]
        return seg_list

    def __charseg(self, query, remove_stopwords=False):
        """Tokenize the sentence into char list.

        Args:
            query: str, input sentence.
            remove_stopwords: bool, whether the sentence needs to remove stopwords. Default: False.

        Returns:
            r: Char list.
        """
        # TODO: fix bug when dealing with English words.
        if remove_stopwords:
            seg_list = ' '.join(jieba.cut(query)).split()
            query = ''.join([x for x in seg_list if x not in self.stopwords])
        query = list(query)
        r = []
        s = ''
        for q in query:
            if q.strip() == '':
                continue
            if q.encode('utf-8').isalnum():
                s += q
            else:
                if s != '':
                    r.append(s)
                    s = ''
                r.append(q)
        if s != '':
            r.append(s)
        return r

    def __ng(self, xlist, n):
        """Combine tokens based on a slide window.

        Args:
            xlist: list, token list.
            n: int, window size.

        Returns:
            ret: list, result token list.
        """
        ret = []
        for i, x in enumerate(xlist):
            diff = i - n + 1
            if diff >= 0:
                tmp = []
                for j in range(n):
                    k = i - j
                    tmp.append(xlist[k])
                tmp.reverse()
                ret.append(''.join(tmp))
        return ret

    def __ngrams(self, xlist, n):
        """Combine tokens based on multi slide windows. The size of windows is not more than max gram size.

        Args:
            xlist: list, token list.
            n: int, max gram size.

        Returns:
            ret: list, result token list.
        """
        ret = []
        for i in range(n):
            ret.extend(self.__ng(xlist, n - i))
        return ret


def cut(self, sentence, cut_all=False, HMM=True):
    """The new cut function to replace `jieba.cut` function.

    The details can be seen at https://yuque.alibaba-inc.com/ivr_algo/business/mc1fdi#6089454d.
    """
    sentence = jieba.strdecode(sentence)

    re_eng_word = jieba.re_eng_word
    if cut_all:
        re_han = jieba.re_han_cut_all
        re_skip = jieba.re_skip_cut_all
    else:
        re_han = jieba.re_han_default
        re_skip = jieba.re_skip_default
    if cut_all:
        cut_block = self._Tokenizer__cut_all
    elif HMM:
        cut_block = self._Tokenizer__cut_DAG
    else:
        cut_block = self._Tokenizer__cut_DAG_NO_HMM
    blocks = re_han.split(sentence)
    for blk in blocks:
        if not blk.strip():
            continue
        if re_eng_word.match(blk):
            yield blk
        elif re_han.match(blk):
            for word in cut_block(blk):
                yield word
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if re_skip.match(x):
                    yield x
                elif not cut_all:
                    for xx in x:
                        yield xx
                else:
                    yield x
