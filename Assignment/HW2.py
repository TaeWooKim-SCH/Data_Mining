import pandas as pd
import numpy as np

# 문제 1. 다음 주어진 코드를 실행해서 나타난 데이터프레임에서의 주대각성분을 모두 0 으로 바꾸세요
    # 방법1: 판다스만 이용해서 풀기
print("방법1: 판다스만 이용해서 풀기")
data = pd.DataFrame(np.random.randint(1,10, 100).reshape(10, -1))
print(data)
print("\n")
for i in range(0, 10):
    data.loc[i][i] = 0
print(data)
print("\n")

    # 방법2: 넘파이를 이용해서 풀기
print("방법2: 넘파이를 이용해서 풀기")
data = pd.DataFrame(np.random.randint(1,10, 100).reshape(10, -1))
print(data)
print("\n")
np.fill_diagonal(data.values, 0) # DataFrame.values = DataFrame.to_numpy()
print(data)



# 문제2. 다음 주어진 코드를 보고 문제에 답하세요
df = pd.DataFrame(np.arange(36).reshape(-1,6), columns=list('abcdef'))
print(df)

    # 1. a와 d 열을 서로 바꾸어서 출력하세요
print("1. a와 d 열을 서로 바꾸어서 출력하세요")
df = pd.DataFrame(np.arange(36).reshape(-1,6), columns=list('abcdef'))
print(df[list('dbcaef')])
print("\n")

    # 2. 열 이름을 하드코딩(hard coding) 하지 않고 일반 함수를 만들어 입력된 두 열을 교환할 수 있도록 구현하세요.
print("2. 열 이름을 하드코딩(hard coding) 하지 않고 일반 함수(예; swap(a,e)는 a과 e열을 서로 교환)를 만들어 입력된 두 열을 교환할 수 있도록 구현하세요.")
df = pd.DataFrame(np.arange(36).reshape(-1,6), columns=list('abcdef'))
def swap_columns(dataframe, c1, c2): # 함수정의
    c_list = list(dataframe.columns)
    c_list[c_list.index(c2)], c_list[c_list.index(c1)] = c_list[c_list.index(c1)], c_list[c_list.index(c2)]
    return dataframe[c_list]
df = swap_columns(df, "a", "d")
print(df)

    # 3. 열을 알파벳 역순으로 정렬합니다. 즉, 'f' 열이 처음부터 'a' 열이 마지막입니다
print("3. 열을 알파벳 역순으로 정렬합니다. 즉, 'f' 열이 처음부터 'a' 열이 마지막입니다")
df = pd.DataFrame(np.arange(36).reshape(-1,6), columns=list('abcdef'))
print(df[reversed(list("abcdef"))])



# 문제 3. 유클리드 거리는 데이터 간의 유사도를 표현하는데 사용합니다. 
    # 다음 주어진 코드에 두 열을 추가해서 나타난 결과처럼 각 행과 가장 가까운 거리의 행을 
    # index 번호로 표시하고 해당 행과의 거리를 나타내도록 코드를 완성하세요.
index_l=[1,2,3,4,5,6,7,8,9,10]
df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1), columns=list('pqrs'), index=index_l)
print(df)
print("")

        # 기본 셋팅
df_array = np.array(df) # 수학 연산을 위해 배열로 바꿔줌.
dist_min_list = [] # 가까운 거리의 행번호의 열과 거리 열을 추가해주기 위한 리스트. 최종 과정에서 배열로 변경.

        # 거리 연산
for i in range(len(df_array)):
    dist_dic = dict() # 가까운 거리의 행번호의 열과 거리열을 추가해주기 위한 딕셔너리.
    for x in range(len(df_array)):
        if i != x: # 자기 자신과 거리를 잴 필요는 없음
            dist = np.sqrt(np.sum((df_array[i, :] - df_array[x, :]) ** 2)) # 두 점의 거리 구하기.
            dist_dic.update({dist:x}) # 거리를 Key로 넣고 그때 인덱스 번호를 Value로 추가.
    dist_min_list.append([dist_dic.get(min(dist_dic)) + 1, min(dist_dic)]) # 데이터 프레임의 행 번호 기준으로 추가해야 되니 인덱스에 1을 더함.

        # 데이터프레임 만들기
dist_array = np.array(dist_min_list, dtype = int) # 배열은 데이터 타입이 한 가지로만 나와야 하기 때문에 index를 추가해주기 위해선 int형으로 배열 생성.
df_array = np.hstack((df_array, dist_array)) # 기존의 df와 행번호의 열과 거리 열을 hstack해줌.
df = pd.DataFrame(df_array, columns = ['p', 'q', 'r', 's', 'nearest_row', 'dist'], index = index_l) # 최종 데이터프레임 생성.
print(df) 



# 문제 4. 한 DataFrame의 null 값을 다른 DataFrame의 null이 아닌 값으로 채워서 두 DataFrame 개체를 결합하여 하나로 나타내는 프로그램을 작성하세요.
df1 = pd.DataFrame({'A': [None, 0, None,4], 'B': [3, 4, 5,6]})
df2 = pd.DataFrame({'A': [1, 1, 1, 3], 'B': [1,2, None, 3]})

print("Original DataFrames:")
print(df1)
print("---------------")
print(df2)
print("---------------")

df1.fillna(df2, inplace = True)
df2.fillna(df1, inplace = True)
print(pd.merge(df1, df2, how = 'outer'))