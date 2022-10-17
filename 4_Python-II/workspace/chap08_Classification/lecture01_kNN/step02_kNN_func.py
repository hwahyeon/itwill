# -*- coding: utf-8 -*-
"""
C:\ITWILL\4_Python-II\workspace\chap08_Classification
"""
# from module import function
from step01_kNN_data import data_set
import numpy as np

# dataset 생성 
know, not_know, cate = data_set() # 알려진 그룹, 알려지지 않은 그룹, 정답값
know.shape # (4, 2) - 알려진 집단 
not_know # array([1.6 , 0.85]) - 알려지지 않은 집단 
cate # array(['A', 'A', 'B', 'B']


# 유클리드 거리계산식 : 차 > 제곱 > 합 > 제곱근
diff =  know - not_know  
diff
square_diff = diff ** 2
square_diff

sum_square_diff = square_diff.sum(axis = 1) # 행 단위 합계 
sum_square_diff # [0.2225, 0.3825, 0.0425, 0.1625]

distance = np.sqrt(sum_square_diff)
distance # [0.47169906, 0.61846584, 0.20615528, 0.40311289] #거리가 가장 가까운 것은 세번째
cate # ['A', 'A', 'B', 'B']

sortDist = distance.argsort() #argsort는 값을 가지고 정렬하는 것이 아니라 내용을 가지고 정렬하는 것
sortDist # [2, 3, 0, 1] # 오름차순으로 정렬하라는 뜻은 거기에 알맞은 인덱스를 붙이라는 것이다.
'''
 즉 [0.47169906, 0.61846584, 0.20615528, 0.40311289]를 오름차순 argsort하면
 [0.20615528, 0.40311289, 0.47169906, 0.61846584]이렇게 하라는 것이 아니라
 [2, 3, 0, 1]작은 순서대로 인덱스를 붙여주는 것.
'''
result = cate[sortDist]
result # ['B', 'B', 'A', 'A']

# k = 3 - 최근접 이웃 3개 
k3 = result[:3] # ['B', 'B', 'A']

# dict 
classify_re = {}

for key in k3 :
    classify_re[key] = classify_re.get(key, 0) + 1
    
classify_re # {'B': 2, 'A': 1}

vote_re = max(classify_re)
print('분류결과 : ', vote_re) # 분류결과 :  B


def knn_classify(know, not_know, cate, k=3) :
    # 단계1 : 거리계산식 
    diff =  know - not_know  
    square_diff = diff ** 2    
    sum_square_diff = square_diff.sum(axis = 1) # 행 단위 합계    
    distance = np.sqrt(sum_square_diff)
    
    # 단계2 : 오름차순 정렬 -> index 
    sortDist = distance.argsort()
    
    # 단계3 : 최근접 이웃(k=3)
    class_result = {} # 빈 set 
    
    for i in range(k) : # k = 3(0~2)
        key = cate[sortDist[i]]
        class_result[key] = class_result.get(key, 0) + 1
        
    return class_result # {'B': 2, 'A': 1}
        
    
class_result = knn_classify(know, not_know, cate)    
  
class_result #  {'B': 2, 'A': 1}

print('분류결과 :', max(class_result, key=class_result.get)) # 분류결과 : B
    
# 영문자의 내림차순에서 B가 나중에 나오기 때문에 max는 B를 반환한다.(value값이 없을때 = key가 생략되었을 때)
# class_result.get해당값의 value값을 가져오는 역할. 즉, 이 식에서는 B가 2개로 가장 많기 때문에
# B가 나오는 것.


'''
class_result #  {'B': 1, 'A': 2}
print('분류결과 :', max(class_result)  # B
print('분류결과 :', max(class_result, key=class_result.get)) # A
'''









