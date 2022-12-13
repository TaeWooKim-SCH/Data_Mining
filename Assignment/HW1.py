# 문제1
import numpy as np

dates = np.arange(np.datetime64('2022-09-01'), np.datetime64('2022-09-27'), 2)
print(dates)

    # P1. (10pt) numpy 함수만을 이용해서 빠진 날짜를 모두 출력하는 코드를 구현하세요.
dates2 = np.arange(np.datetime64('2022-09-02'), np.datetime64('2022-09-27'), 2)
print(dates2)

    # P2. (10pt) for 구문을 이용해서 빠진 날자를 모두 출력하는 코드를 구현하세요.
        # 방법1: 리스트 이용
missing_date = []
for i in range(2, 27, 2):
    missing_date.append(np.datetime64('2022-09-{}'.format(str(i).zfill(2)))) # zfill(2): 1자리 숫자일때 0을 추가해 2자리수로 만들어줌. sting형일 때 가능
print(np.array(missing_date))

        # 방법2: 배열만 이용
missing_date =  np.arange(np.datetime64('2022-09-01'), np.datetime64('2022-09-27'), 2)
for i in range(2, 27, 2):
    missing_date = np.insert(missing_date, i - 1, np.datetime64('2022-09-{}'.format(str(i).zfill(2))))
missing_date = missing_date.reshape(-1, 2)
print(missing_date[:, 1])

        # 방법3: 배열만 이용
print("P2 - 방법3")
missing_date = np.zeros(13, dtype = "datetime64[D]")
x = 2
for i in range(13):
    missing_date[i] = np.datetime64('2022-09-{}'.format(str(x).zfill(2)))
    x += 2
print(missing_date)



# 문제2
import numpy as np

phoenix= np.array("I'm a SCH student and I'm OK and I study all day".split(" "))
print(phoenix)
search_results, = np.where(np.char.str_len(phoenix) >=5)
print(search_results)
print(phoenix[search_results])

# P3. (10pt) 위 코드의 실행 결과로 얻은 부분 집합 요소인 문자열을
    # 1) 첫글자만 대문자로 변경하세요.
title_phoenix = phoenix[search_results]
for i in range(len(title_phoenix)):
    title_phoenix[i] = title_phoenix[i].title()
print(title_phoenix)

    #  2) 모두 대문자로 변경하세요.
upper_phoenix = phoenix[search_results]
for i in range(len(upper_phoenix)):
    upper_phoenix[i] = upper_phoenix[i].upper()
print(upper_phoenix)

# P4. (10Pt) 어떻게 phoenix 배열에 있는 각 문자열을 시작과 끝에 별문자(*)로 묶을 수 있습니까? 코드로 구현하세요.
    # 방법1: 리스트 이용 - for문을 사용해 배열의 원소 양 끝에 "*"를 추가해주면 된다.
print("방법1")
phoenix= np.array("I'm a SCH student and I'm OK and I study all day".split(" "))
phoenix = list(phoenix) # array로 그대로 갖고 가면 student에서 끝에 별이 누락된다. 이유를 찾아봤지만 찾을 수 없었다.
for i in range(len(phoenix)):
    phoenix[i] = "*" + phoenix[i] + "*"
print(np.array(phoenix))

    # 방법2: 배열만 이용 - '*' 배열을 만들어줘서 phoenix 배열과 더해준다.
print("방법2")
phoenix= np.array("I'm a SCH student and I'm OK and I study all day".split(" "))
star = np.array(["*"] * len(phoenix))
star_phoenix = np.char.add(star, phoenix)
star_phoenix = np.char.add(star_phoenix, star)
print(star_phoenix)





