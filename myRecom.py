import os
import pandas as pd
import numpy as np
from sklearn.decomposition import randomized_svd, non_negative_factorization

#평가지표
def RMSE(y_true, y_pred) : 
    return np.sqrt(np.mean(np.array(y_true)-np.array(y_pred))**2)

def adjacentRow(_data, type = 1) : 
    '''
    type값에 따른 인접행렬을 생성 해 줍니다. 
    
    입력값 
    _data : raw_data
    type = 1 : 연결성이 있는 자리에  1로 계산
    type = else : 연결성이 있는 자리에  rating 값으로 계산
    
    반환값 
    adj_matrix : 인접행렬
    
    '''
    raw_data = _data.copy()
    
    # id가 배열인덱스와 일치하도록 -> 0부터 시작할 수 있게 조정 --> 각 id를 행x열로 표시하기 위해
    raw_data[:,0] -= 1 # user_id
    raw_data[:,1] -= 1 # movie_id

    # 인접행렬의 크기 계산 
    n_users = np.max(raw_data[:, 0])
    n_movies = np.max(raw_data[:,1] )
    shape = (n_users+1, n_movies+1)

    # 인접행렬 생성
    # user_id - movie_id 와의 관계를 matrix로 표현
    # 1 : 평가 데이터 있음 
    adj_matrix = np.ndarray(shape, dtype=int)
    
    if type != 1: 
        for user_id, movie_id, rating, time in raw_data:
            adj_matrix[user_id][movie_id] = rating
    else :     
        for user_id, movie_id, rating, time in raw_data:
            adj_matrix[user_id][movie_id] = 1 

    return adj_matrix

def compute_cos_similarity(vector_origin , vector_target) : 
    '''
    코사인 유사도를 계산합니다.
    
    입력값
    vector_origin : 기준이 되는 벡터
    vector_target : 비교할 벡터
    
    반환값
    dot/(norm1*norm2) : 코사인유사도 계산값
    
    '''
    norm1 = np.sqrt(np.sum(np.square(vector_origin))) #벡터크기 계산
    norm2 = np.sqrt(np.sum(np.square(vector_target)))
    dot = np.dot(vector_origin,vector_target)
    
    return dot/(norm1*norm2)

def calculSimilarity (id, vector , _matrix, type = 'dot') : 
    '''
    타입별 유사도를 계산 해 줍니다.
    
    입력값
    id : my_id
    vector : my_vector
    _matrix : adj_matrix
    type  : 유사도 계산 방식
        'dot' : 내적곱
        'euclidean' : 유클리드거리 방식
        'cos' : 코사인유사도 
    
    반환값
    best_match : 가장 높은 유사도값
    user_id : my_id와 유사한 id 값
    user_vector : user_id 의 vector 값
    '''
    adj_matrix = _matrix.copy()
    best_match , best_match_id, best_match_vector = -1, -1, []
    
    #user_vector : 사용자가 평가한 movie_id 벡터 
    for user_id, user_vector in enumerate(adj_matrix) : 
        
        if id != user_id: # 나 자신이 아닌 다른 사용자와 값 비교
            
            if type == 'cos' : 
                similarity = compute_cos_similarity(vector,user_vector) 
                
                if similarity > best_match : 
                    best_match = similarity
                    best_match_id = user_id
                    best_match_vector = user_vector
                
                
            if type == 'dot' : 
                similarity = np.dot(vector, user_vector) 
                
                if similarity > best_match : 
                    best_match = similarity
                    best_match_id = user_id
                    best_match_vector = user_vector
                
            if type == 'euclidean' : 
                similarity = np.sqrt(sum(np.square(vector-user_vector))) 
                best_match = 9999
                
                if similarity < best_match : 
                    best_match = similarity
                    best_match_id = user_id
                    best_match_vector = user_vector
                
    print('Best Match similarity: {}, Best Match ID : {}'.format(best_match, best_match_id))  
              
    return best_match, best_match_id , best_match_vector

def recommendVector(vector1, vector2) : 
    '''
    두 백터를 비교하여, 중복되지 않는 요소리스트를 추출해 줍니다.
    
    입력값
    vector1 : 기준 백터
    vector2 : 대조 백터터
    
    반환값
    recommend_list : 추천 리스트
    '''
    # 추천 리스트 뽑기 
    recommend_list = []
    for i , log in enumerate(zip(vector1, vector2)) : 
        log1, log2 = log
        if log1 < 1. and log2 > 0. : #log1 < 1 -> 내가 보지 않은 영화 , log2 > 0 : 다른 유저가 본 영화
            recommend_list.append(i)

    print('best_match movies : ', recommend_list)
    
    return recommend_list
    
    
def dotProductRecommend(_matrix, my_id = 0) : 
    '''
    내적을 통해 추천목록을 생성해 줍니다.
    
    입력값
    _matrix : adj_matrix
    my_id = 사용자 id값
    
    반환값
    best_match : 가장 높은 유사도값
    user_id : my_id와 유사한 id 값
    user_vector : user_id 의 vector 값
    '''
    adj_matrix = _matrix.copy()
    
    # my_id 백터값 생성
    my_vector = adj_matrix[my_id]

    # my_id 와 유사도가 높은 user의 vector 추출
    best_match, best_match_id, best_match_vector = calculSimilarity (my_id, my_vector , adj_matrix, type = 'dot')
    
    # 추천리스트 추출
    recommend_list = recommendVector(my_vector, best_match_vector)
    
    return best_match, best_match_id, recommend_list


def EuclideanRcommend(_matrix, my_id = 0) : 
    '''
    유클리드 거리공식 기반 유사도 계산을을 통해 추천목록을 생성해 줍니다.
    
    입력값
    _matrix : adj_matrix
    my_id = 사용자 id값
    
    반환값
    best_match : 가장 높은 유사도값
    user_id : my_id와 유사한 id 값
    user_vector : user_id 의 vector 값
    '''
    adj_matrix = _matrix.copy()

    # 테스트값 생성
    my_id, my_vector =my_id, adj_matrix[my_id]
    
    # my_id 와 유사도가 높은 user의 vector 추출
    best_match, best_match_id, best_match_vector = calculSimilarity (my_id, my_vector , adj_matrix, type = 'euclidean')
    
    # 추천 리스트 뽑기 
    recommend_list = recommendVector(my_vector, best_match_vector)
    
    return best_match, best_match_id, recommend_list



def cosRecommend(_matrix, my_id = 0) : 
    '''
    코사인 유사도 계산을을 통해 추천목록을 생성해 줍니다.
    
    입력값
    _matrix : adj_matrix
    my_id = 사용자 id값
    
    반환값
    best_match : 가장 높은 유사도값
    user_id : my_id와 유사한 id 값
    user_vector : user_id 의 vector 값
    '''
    adj_matrix = _matrix.copy()

    # 테스트값 생성
    my_id, my_vector =  my_id, adj_matrix[my_id]
    
    # my_id 와 유사도가 높은 user의 vector 추출
    best_match, best_match_id, best_match_vector = calculSimilarity (my_id, my_vector , adj_matrix, type = 'cos')
    
    # 추천 리스트 뽑기 
    recommend_list = recommendVector(my_vector, best_match_vector)
    
    return best_match , best_match_id , recommend_list




def hybridRecommend (_matrix, i=0 , embediing ='svd' ,similarity ='cos', type = 'user') : 
    '''
    svd 임베딩 후 코사인 유사도 계산을을 통해 추천목록을 생성해 줍니다.
    
    입력값
    _matrix : adj_matrix
    i : 사용자 id값
    type ='user' : 사용자 기반 추천 목록 생성 
    type = 'item' : 아이템 기반 추천 목록 생성 
    
    반환값
    best_match : 가장 높은 유사도값
    user_id : my_id와 유사한 id 값
    user_vector : user_id 의 vector 값
    '''
    adj_matrix = _matrix.copy()
    
    if embediing == 'svd':
        U, S, V = randomized_svd(adj_matrix, n_components=2)
        
        user_matrix = U
        item_metrix = V.T
        
    if embediing =='nmf' :
        W, H, iter = non_negative_factorization(adj_matrix, n_components=2)

        user_matrix = W
        item_metrix = H.T
        
    # 테스트값 생성
    if type == 'user' : 
        
        my_id, my_vector = i, user_matrix[i] 
        best_match, best_match_id, best_match_vector  = calculSimilarity (my_id, my_vector , user_matrix, type = similarity) 
        
        # 추천 리스트 뽑기 
        recommend_list = recommendVector(adj_matrix[my_id], adj_matrix[best_match_id])
                
    if type == 'item' :
        
        my_id, my_vector = i, item_metrix[i] 
        best_match, best_match_id, best_match_vector  = calculSimilarity (my_id, my_vector , item_metrix, type = similarity) 
        
        # 추천 리스트 뽑기 
        recommend_list = []
        
        for i , user_vector in enumerate(zip(adj_matrix)) : 
            if adj_matrix[i][my_id]>0.9 :
                recommend_list.append(i)
    
        print('best_match movies : ', recommend_list)
    
    return best_match, best_match_id, recommend_list