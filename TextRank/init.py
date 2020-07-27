# coding: utf-8
from textrank import TextRank #textrank 모듈 불러오기

f = open("text.txt", 'r', encoding='utf-8') #stopwords 템플릿
text = f.read()
tr = TextRank(text)    #textrank 실행
f.close()
i = 1
for row in tr.summarize(3):    #요약된 문장과 키워드 출력
    print(str(i)+'. '+row)
    i += 1
print('keywords :', tr.keywords())