import numpy as np

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
