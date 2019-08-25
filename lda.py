# coding:utf-8

from numpy import *
import jieba
import suggest_freq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDA():
    '''
    1.读取3个文件；
    2.对三个文件进行分词处理；
    3.引入停用词；
    4.LDA模型建立；
    '''

    def loadfile(self, filename):
        '''
        1.读取文件
        2.分词
        '''
        with open(filename, 'rb') as file:
            doc = file.read()
            docd = doc.decode('gbk')
            # jieba分词
            document = jieba.cut(docd)
            # 显示分析的结果
            self.doc_result = ' '.join(document)
        # print(self.doc_result)
        return self.doc_result

    def stop_word(self):
        '''
        3.引入停用词
        '''
        text = open('stop_words.txt')
        self.words_stop = text.read()
        # print(self.words_stop)
        return self.words_stop

    def lda_model(self):
        '''
        4.模型建立：基于词频统计的LDA模型
        '''
        result00 = self.loadfile('nlp_test0.txt')
        result02 = self.loadfile('nlp_test2.txt')
        result04 = self.loadfile('nlp_test4.txt')
        all_array = [result00, result02, result04]
        # 去停用词
        stop_words = self.stop_word()
        vector = CountVectorizer(stop_words=list(stop_words))
        vector_words = vector.fit_transform(all_array)
        # 词表
        word_list = vector.get_feature_names()
        # 计算lda值--主题设为K=2
        lda = LatentDirichletAllocation(n_topics=2, learning_offset=50, random_state=0)
        # 主题分布
        print('*****************************主题分布*******************************')
        ZT_FB = lda.fit_transform(vector_words)
        print(ZT_FB)
        # 词分布
        print('******************************词分布********************************')
        C_FB = lda.components_
        print(C_FB)
        print(shape(C_FB))


    def Observed_CB(self):
        '''
        注：观察每个文档去停用词后的词表
        '''
        result00 = self.loadfile('nlp_test0.txt')
        result02 = self.loadfile('nlp_test2.txt')
        result04 = self.loadfile('nlp_test4.txt')
        # all_array = [result00, result02, result04]
        stop_word = self.stop_word()
        vec = CountVectorizer(stop_words=list(stop_word))
        # 词表01
        vec.fit_transform([result00])
        CB00_result = vec.get_feature_names()
        print('******每个文档词表个数******')
        print('CB00_result:', len(CB00_result), CB00_result)
        # 词表02
        vec.fit_transform([result02])
        CB02_result = vec.get_feature_names()
        print('CB02_result:', len(CB02_result), CB02_result)
        # 词表04
        vec.fit_transform([result04])
        CB04_result = vec.get_feature_names()
        print('CB04_result:', len(CB04_result), CB04_result)
        # print('****************************')


# 主函数
if __name__ == '__main__':
    lda = LDA()
    lda.Observed_CB()
    lda.lda_model()