import numpy as np
import pandas as pd
import os

# 판다스란?
    # 시리즈 객체
list_data = [1, 2, 3, 4, 5]
list_name = ["a", "b", "c", "d", "e"]
example_obj = pd.Series(data = list_data, index = list_name)
print(example_obj)
print(example_obj.index)
print(example_obj.values)
print(example_obj.dtype)

example_obj.name = "number"
example_obj.index.name = "id"
print(example_obj)

    # 시리즈 객체 생성하기
dict_data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
example_obj = pd.Series(data = dict_data, dtype = np.float32, name = "example_data")
print(example_obj)

    # 판다스의 모든 객체는 인덱스 값을 기준으로 생성
        # 기존 데이터에 인덱스 값을 추가하면 NaN 값이 출력됨
dict_data_1 = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
indexes = ["a", "b", "c", "d", "e", "f", "g", "h"]
series_obj_1 = pd.Series(dict_data_1, index = indexes)
print(series_obj_1)

    # 데이터프레임의 생성
        # 'read_확장자'함수로 데이터 바로 로딩
            # .csv나 .xlsx 등 스프레드시트형 확장자 파일에서 데이터 로딩
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data' # 데이터 URL을 변수 data_url에 넣기
df_data = pd.read_csv(data_url, sep = '\s+', header = None) # csv 데이터 로드
df = pd.DataFrame(df_data) # 데이터프레임으로 만들기
print(df)

    # 데이터프레임 직접 만들기
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 
            'age': [42, 52, 36, 24, 73], 
            'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])
print(df)

    # 데이터프레임의 열 다루기
print(pd.DataFrame(raw_data, columns = ['age', 'city']))
print(pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city', 'debt']))



# 데이터 추출
    # 데이터 로딩
df = pd.read_excel("G:\내 드라이브\김태우\대학교\\2학년 2학기\데이터마이닝\Ex_Data\ch04\excel-comp-data.xlsx")

    # 열 이름을 사용한 데이터 추출
print(df.head(5))
print(df.head(3).T)
print(df[["account", "street", "state"]].head(3))
    # 행 번호를 사용한 데이터 추출
print(df[:3])

    # 행과 열을 모두 사용한 데이터 추출
print(df[["name", "street"]][:2])
df.index = df["account"] # 기존 df 테이블의 인덱스를 account 열로 변경
del df["account"] # account 열은 중복되니 제거
print(df.head())
print(df.loc[[211829, 320563], ['name', 'street']]) # 두 개의 인덱스 행에서 name과 street열 추출
print(df.loc[205217:, ['name', 'street']])
print(df.iloc[:10, :3])

    # loc, iloc 함수를 사용한 데이터 추출
df_new = df.reset_index() # 데이터 테이블 초기화
print(df_new)

    # drop 함수
print(df_new.drop(1).head()) # 인덱스 1번 째 행 삭제
df_drop = df_new.drop(1) # drop은 메서드. 내장함수가 아니기 때문에 값저장 x
print(df_drop)
print(df_new.drop(1, inplace = True)) # 내장함수처럼 사용하기 위해선 inplace를 사용해 원본 객체에 영향을 줌
print(df_new)
print(df_new.drop('account', axis = 1)) # account 열 제거
print(df_new.drop(["account", "name"], axis = 1)) # account, name 열 제거



# 그룹별 집계
    # 그룹별 집계 사용하기
        # 그룹별 집계의 기본형
ipl_team = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings', 'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'], 
            'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2], 
            'Year': [2014, 2015, 2014, 2015, 2015, 2015, 2016, 2017, 2016, 2014, 2015, 2017], 
            'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
df = pd.DataFrame(ipl_team)
print(df)
print(df.groupby('Team')['Points'].sum())

        # 멀티 인덱스 그룹별 집계
multi_gropby = df.groupby(['Team', 'Year'])['Points'].sum()
print(multi_gropby)

        # 멀티 인덱스
multi_gropby = df.groupby(['Team', 'Year'])['Points'].sum()
print(multi_gropby.index)
print(multi_gropby['Devils':'Riders'])
print(multi_gropby.unstack()) 
print(multi_gropby.swaplevel().sort_index())
print(multi_gropby.sum(level = 0))
print(multi_gropby.sum(level = 1))


    # 그룹화된 상태
grouped = df.groupby('Team')
print(grouped.get_group('Riders'))
        # 집계
print(grouped.agg(min))
print(grouped.agg(np.mean))

        # 변환
print(grouped.transform(max))
score = lambda x: (x - x.mean()) / x.std()
print(grouped.transform(score))

        # 필터
print(df.groupby('Team').filter(
    lambda x: len(x) >= 3
))
print(df.groupby('Team').filter(
    lambda x: x['Points'].max() > 800
))



# 병합과 연결
    # 병합
        # 내부 조인
raw_data = {
            'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
            'test_score': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]
            }
df_left = pd.DataFrame(raw_data, columns = ['subject_id', 'test_score'])
print(df_left)

raw_data = {
            'subject_id': ['4', '5', '6', '7', '8'],
            'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
            'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']
            }
df_right = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
print(df_right)

print(pd.merge(left = df_left, right = df_right, how = "inner", on = 'subject_id'))

        # 왼쪽 조인, 오른쪽 조인
print(pd.merge(df_left, df_right, on = 'subject_id', how = 'left')) # 왼쪽 조인
print(pd.merge(df_left, df_right, on = 'subject_id', how = 'right')) # 오른쪽 조인

        # 완전 조인
print(pd.merge(df_left, df_right, on = 'subject_id', how = 'outer'))


    # 연결
filenames = [os.path.join("G:\내 드라이브\김태우\대학교\\2학년 2학기\데이터마이닝\Ex_Data\ch04", filename)
                for filename in os.listdir("G:\내 드라이브\김태우\대학교\\2학년 2학기\데이터마이닝\Ex_Data\ch04")
                if "sales" in filename]
print(filenames)

        # concat 사용
df_list = [pd.read_excel(filename, engine = "openpyxl")
            for filename in filenames]
for df in df_list:
    print(type(df), len(df))

df = pd.concat(df_list, axis = 0)
print(df)
print(df.reset_index(drop = True))

        # append 사용: append 함수는 파일을 한 개씩 합치기 때문에 두 개 이상의 데이터프레임을 합칠 때에는 concat 함수를 쓰는 것이 좋다.
df_1, df_2, df_3 = [pd.read_excel(filename, engine = "openpyxl")
                        for filename in filenames]
df = df_1.append(df_2)
df = df.append(df_3)
print(df)


