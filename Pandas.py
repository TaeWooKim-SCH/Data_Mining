import numpy as np
import pandas as pd

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


