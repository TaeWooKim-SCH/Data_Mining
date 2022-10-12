import numpy as np
import sys

# 넘파이 배열 객체 다루기
    #  배열의 생성
test_array = np.array([1, 4, 5, 8], float)
print(test_array)

a = np.array([1, 2, 3, '4'])
print(a) # 넘파이 배열은 문자와 숫자가 섞이면 모두 문자로 취급

test_array = np.array([1, 4, 5, "8"], float)
print(type(test_array[3])) # 실수형으로 자동 형 변환 실시

print(test_array.dtype) # 배열 전체의 데이터 타입 반환
print(test_array.shape) # 배열의 구조(shape)를 반환


    # 배열의 구조
matrix = [[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]]
print(np.array(matrix, int).shape)

tensor_rank3 = [
    [[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]],
    [[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]],
    [[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]],
    [[1, 2, 5, 8], [1, 2, 5, 8], [1, 2, 5, 8]]
]
print(np.array(tensor_rank3, int).shape)
print(np.array(tensor_rank3, int))
print(np.array(tensor_rank3, int).ndim) # ndim: 배열의 차원 수
print(np.array(tensor_rank3, int).size) # size: 전체 원소의 갯수 반환


    # dtype
print(np.array([[1, 2, 3.5], [4, 5, 6.5]], dtype = int)) # 반올림이 아닌 앞의 정수만 반환
print(np.array([[1, 2, 3.5], [4, 5, 6.5]], dtype = float))

print(np.array([[2, 3.5], [5, 6.5]], dtype = np.float64).itemsize) # 원소의 개수가 아닌 배열의 각 요소가 차지하는 바이트 확인


    # 배열의 구조 다루기
x = np.array([[1, 2, 5, 8], [1, 2, 5, 8]])
print(x.shape)
print(x.reshape(-1, )) # -1하나만 행 위치에 들어간 것은 1차원 배열을 만들어주는 것
print(x.reshape(1, -1)) # 모양으로 따지면 위와 같지만, 이것은 2차원 배열

x = np.array(range(8))
print(x)

x = np.array(range(8)).reshape(4, 2)
print(x)
print(x.reshape(2, -1))
print(x.reshape(2, 2, -1)) # 2층, 2행, 열은 앞에 값에 따라 자동 반환

x = np.array(range(8)).reshape(2, 2, 2)
print(x)
print(x.flatten()) # flatten 함수는 데이터 그대로 1차원으로 변경


    # 인덱싱
x = np.array([[1, 2, 3], [4, 5, 6]], int)
print(x)
print(x[0][0])
print(x[0, 2])
x[0, 1] = 100 # 3차원 인덱싱부터는 reshape 함수 입력 값 순서와 같음
print(x)
a = np.array(range(12)).reshape(2, 2, -1) # 3차원 배열
print(a)
print(a[1, 1, 2])

    # 슬라이싱
x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], int)
print(x[:, 2:]) # 전체 행의 3열 이상
print(x[1, 1:3]) # 2행 2열부터 3열까지
print(x[1:3]) # 2행~3행의 전체. 하지만 2행까지만 존재하기 때문에 초과한 인덱스는 무시

x = np.array(range(15), int).reshape(3, -1)
print(x)
print(x[:, ::2]) # 증가값: [시작 인덱스:마지막 인뎃스:증가값]
print(x[::2, ::3])


    # arange: (시작 인덱스, 마지막 인데스, 증가값)
print(np.arange(10)) 
print(np.arange(-5, 5))
print(np.arange(0, 5, 0.5))


    # ones, zeros, empty
print(np.ones(shape = (5, 2), dtype = np.int8)) # ones: 1로만 구성된 넘파이 배열 생성
print(np.zeros(shape = (2, 2), dtype = np.float32)) # zeros: 0으로만 구성된 넘파이 배열 생성
print(np.empty(shape = (2, 4), dtype = np.float32)) # empty: 활용 가능한 메모리 공간 확보하여 반환


    # ones_like, zeros_like, empty_like
x = np.arange(12).reshape(3, 4)
print(x)
print(np.ones_like(x)) # ones_like: 기존 넘파이 배열과 같은 크기로 만들어 내용을 1로 채움
print(np.zeros_like(x)) # zeros_like: 기존 넘파이 배열과 같은 크기로 만들어 내용을 0으로 채움
print(np.empty_like(x)) # empty_like: 기존 넘파이 배열과 같은 크기로 만들어 빈 상태로 만듦


    # identity, eye, diag
        # identity: 단위행렬 생성
print(np.identity(n = 3, dtype = int))
print(np.identity(n = 4, dtype = int))

        # eye: 시작점과 행렬 크기를 지정, 단위행렬 생성
print(np.eye(N = 3, M = 5)) # N: 행의 개수, M: 열의 개수
print(np.eye(N = 3, M = 5, k = 2)) # k: 열의 값을 기준으로 시작 인덱스

        # diag: 행렬의 대각성분 값을 추출
matrix = np.arange(9).reshape(3, 3)
print(matrix)
print(np.diag(matrix))
print(np.diag(matrix, k = 1)) # k: 열의 값을 기준으로 시작 인덱스


    # 통계 분석 함수   
print(np.random.uniform(0, 5, 10)) # uniform(균등분포 함수): 시작값, 끝 값, 데이터 개수
print(np.random.normal(0, 2, 10)) # normal(정규분포 함수): 평균값, 분산, 데이터 개수