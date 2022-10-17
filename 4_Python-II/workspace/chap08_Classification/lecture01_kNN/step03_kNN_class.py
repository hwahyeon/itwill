# -*- coding: utf-8 -*-

# from module import function
from step01_kNN_data import data_set
import numpy as np

# dataset 
know, not_know, cate = data_set()

class kNNClassify :
    # 생성자, 멤버(메서드, 변수)
    
    def classify(self, know, not_know, cate, k=3) :
        # 단계1 : 거리계산식 
        diff =  know - not_know  
        square_diff = diff ** 2    
        sum_square_diff = square_diff.sum(axis = 1) # 행 단위 합계    
        distance = np.sqrt(sum_square_diff)
        
        # 단계2 : 오름차순 정렬 -> index 
        sortDist = distance.argsort()
        
        # 단계3 : 최근접 이웃(k=3)
        self.class_result = {} # 멤버 변수 
        
        for i in range(k) : # k = 3(0~2)
            key = cate[sortDist[i]]
            self.class_result[key] = self.class_result.get(key, 0) + 1
            
    def vote(self) :
        vote_re = max(self.class_result, key=self.class_result.get)
        print('분류결과 :', vote_re)
    
# 객체 생성 : 생성자 이용 
knn = kNNClassify()
knn.classify(know, not_know, cate) # class_result 생성 
knn.class_result # {'B': 2, 'A': 1}
knn.vote() # 분류결과 : B

        
    


