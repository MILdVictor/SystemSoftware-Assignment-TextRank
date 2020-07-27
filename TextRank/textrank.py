
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF 모델 생성을 위한 머신러닝 패키지
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer




class SentenceTokenizer(object):    #원문의 문장화 및 명사 추출 클래스

    def splitsenteces(self, text):  #원문을 문장 단위로 분리
        tokens = [word for word in nltk.sent_tokenize(text)]
        return tokens

    def preprocessing(self, sentences):     #문장을 전처리하여 일반화시킴
        proc_words = []
        for sentence in sentences:
            #단어화
            tokens = [word for word in nltk.word_tokenize(sentence)]
            # 전체 소문자화
            tokens = [word.lower() for word in tokens]
            #불용어 처리
            stop = stopwords.words('english')
            stop.append('the')
            tokens = [token for token in tokens if token not in stop]
            #3자 이하의 단어 삭제
            tokens = [word for word in tokens if len(word) >= 3]

            #표제어 추출
            lmtzr = WordNetLemmatizer()
            tokens = [lmtzr.lemmatize(word) for word in tokens]
            #동사 표제화
            tokens = [lmtzr.lemmatize(word, 'v') for word in tokens]

            sent = ' '.join(tokens)
            proc_words.append(sent)

        return proc_words





class GraphMatrix(object):          #TF-IDF 모델 그래프 생성 클래스
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []

    def build_sent_graph(self, sentence):   #각 문장간의 correlation matrix를 이용한 가중치 그래프 생성
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)

        return self.graph_sentence

    def build_words_graph(self, sentence):  #각 단어간의 correlation matrix를 이용한 가중치 그래프 생성
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_

        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]: word for word in vocab}


                             #문장과 단어의 순위 계산 클래스
class Rank(object):                     #TR(V(i)) = (1-d) + d*sum(in(V(i)))(w(j,i)*TR(V(j))/sum(out(V(j)))(w(j,k)))
    def get_ranks(self, graph, d=0.85):  # d = damping factor (구글 pagerank에서도 0.85를 사용)
        A = graph
        matrix_size = A.shape[0]

        for id in range(matrix_size):
            A[id, id] = 0  # 그래프의 (n,n) 부분을 0으로
            link_sum = np.sum(A[:, id])  # A[:, id] = A[:][id]

            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1

        B = (1 - d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)  # 연립방정식 Ax = b

        return {idx: r[0] for idx, r in enumerate(ranks)}   #idx : rank 형태의 dictionary 반환




class TextRank(object):         #위의 클래스들을 전체적으로 구동하는 TextRank 클래스 생성
    def __init__(self, text, method='s', line_split=True):  #매개변수로 원문, 불용어, 줄바꿈 문장 구분 여부를 받는다
        self.sent_tokenize = SentenceTokenizer()
        self.sentences = self.sent_tokenize.splitsenteces(text)
        self.words = self.sent_tokenize.preprocessing(self.sentences)       #원문에서 분리한 문장들을 다시 단어 단위로 분리한다
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.words)    #단어들로부터 sentence 그래프를 생성한다
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.words)   #단어들로부터 word 그래프와 word dictionary를 생성한다
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)       #sentence 그래프를 이용해 문장의 textrank 가중치를 적용한다
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True) #가중치에 따라 중요문장의 index를 정렬한다
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)      #word 그래프를 이용해 단어의 textrank 가중치를 적용한다
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)   #가중치에 따라 중요 단어의 index를 정렬한다

    def debug(self):
        return self.sent_rank_idx, self.word_rank_idx, self.sent_graph, self.words_graph, self.sentences

    def summarize(self, sent_num=3):        #정렬된 문장들로부터 3개를 기본으로 선택하는 함수
        summary = []
        index = []

        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)

        index.sort()
        for idx in index:                   #최고 우선순위 3개의 index로부터 문장을 찾아 summery에 넣는다
            summary.append(self.sentences[idx])
        return summary

    def keywords(self, word_num=10):        #정렬된 단어들로부터 10개를 기본으로 선택하는 함수
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []
        index = []

        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        # index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])

        return keywords

