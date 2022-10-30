import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 맷플롯립의 구조
    # 파이플롯
X = range(100)
Y = range(100)
plt.plot(X, Y)
plt.show()

X_1 = range(100)
Y_1 = [np.cos(value) for value in X]

X_2 = range(100)
Y_2 = [np.sin(value) for value in X]

plt.plot(X_1, Y_1)
plt.plot(X_2, Y_2)
plt.show()


    # 그림과 축
fig, ax = plt.subplots() # figure와 axes 객체 할당

X_1 = range(100)
Y_1 = [np.cos(value) for value in range(100)]

X_2 = range(100)
Y_2 = [np.sin(value) for value in range(100)]

ax.plot(X_1, Y_1) # plot 함수를 사용하여 그래프 생성
ax.set(title = 'cos graph', # 그래프 제목 
        xlabel = 'X', # X축 라벨
        ylabel = 'Y') # Y축 라벨
plt.show()

fig = plt.figure() # figure 반환
fig.set_size_inches(10, 10) # figure의 크기 지정

ax_1 = fig.add_subplot(1, 2, 1) # 첫 번째 그래프 생성
ax_2 = fig.add_subplot(1, 2, 2) # 두 번째 그래프 생성

ax_1.plot(X_1, Y_1, c = "b") # 첫 번째 그래프 설정
ax_2.plot(X_2, Y_2, c = "g") # 두 번째 그래프 설정
plt.show()


    # 서브플롯 행렬
fig, ax = plt.subplots(nrows = 1, ncols = 2)
plt.show()
print(type(ax))

x = np.linspace(-1, 1, 100) # x 값과 y_n 값 생성. -1부터 1까지 100개로 나누어 공간을 만듦.
y_1 = np.sin(x)
y_2 = np.cos(x)
y_3 = np.tan(x)
y_4 = np.exp(x) # np.exp: 각 요소에 자연상수를 밑으로 하는 지수함수 값 반환

fig, ax = plt.subplots(2, 2) # 2x2 figure 객체를 생성

ax[0, 0].plot(x, y_1) # 첫 번째 그래프 생성
ax[0, 1].plot(x, y_2) # 두 번째 그래프 생성
ax[1, 0].plot(x, y_3) # 세 번째 그래프 생성
ax[1, 1].plot(x, y_4) # 네 번째 그래프 생성

plt.show()

ax1 = plt.subplot(321) # 첫 번째 공간에 axes 생성
plt.plot(x, y_1)
ax2 = plt.subplot(322) # 두 번째 공간에 axes 생성
plt.plot(x, y_1)
ax3 = plt.subplot(312) # 두 번째 공간에 axes 생성
plt.plot(x, y_1)
ax4 = plt.subplot(325) # 다섯 번째 공간에 axes 생성
plt.plot(x, y_1)
ax5 = plt.subplot(326) # 여섯 번째 공간에 axes 생성
plt.plot(x, y_1)

plt.show()



# 맷플롯립으로 그래프 꾸미기
    # 색상
X = range(100)
Y = range(100, 200)
X_1 = range(100)
Y_1 = [value for value in X]

X_2 = range(100)
Y_2 = [value for value in Y]

plt.plot(X_1, Y_1, color = "#000000")
plt.plot(X_2, Y_2, c = "c")

plt.show()


    # 선의 형태
plt.plot(X_1, Y_1, c = "b", linestyle = "dashed")
plt.plot(X_2, Y_2, c = "r", ls = "dotted")

plt.show()


    # 제목
plt.plot(X_1, Y_1, c = "b", linestyle = "dashed")
plt.plot(X_2, Y_2, c = "r", ls = "dotted")

plt.title("Two lines")
plt.show()

fig = plt.figure()
fig.set_size_inches(10, 10)

ax_1 = fig.add_subplot(1, 2, 1)
ax_2 = fig.add_subplot(1, 2, 2)

ax_1.plot(X_1, Y_1, c = "b")
ax_1.set_title("Figure 1")
ax_2.plot(X_2, Y_2, c = "g")
ax_2.set_title("Figure 2")

plt.show()


    # 범례
        # 축 객체마다 범례를 설정할 수 있음
        # legend 함수 사용하여 생성
plt.plot(
    X_1, Y_1,
    c = "b", ls = "dashed",
    label = "line_1"
)
plt.plot(
    X_2, Y_2,
    c = "r", ls = "dotted",
    label = "line_2"
)
plt.legend(
    shadow = True,
    fancybox = False,
    loc = "upper right"
)

plt.title('$y = ax + b$')
plt.xlabel('x_line')
plt.ylabel('y_line')

plt.show()



# 맷플롯립에서 사용하는 그래프
    # 산점도
data_1 = np.random.rand(512, 2)
data_2 = np.random.rand(512, 2)

plt.scatter(
    data_1[:, 0],
    data_1[:, 1],
    c = "b", marker = "x"
)
plt.scatter(
    data_2[:, 0],
    data_2[:, 1],
    c = "r", marker = "o"
)

plt.show()

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N)) ** 2
plt.scatter(x, y, s = area, c = colors, alpha = 0.5)

plt.show()


    # 막대그래프
        # 데이터 생성
data = [[5., 25., 50., 20.],
        [4., 23., 51., 17.],
        [6., 22., 52., 19.]]

        # X좌표 시작점
X = np.arange(0, 8, 2) 

        # 3개의 막대그래프 생성
plt.bar(X + 0.00, data[0], color = 'b', width = 0.50)
plt.bar(X + 0.50, data[1], color = 'g', width = 0.50)
plt.bar(X + 1.0, data[2], color = 'r', width = 0.50)

        # X축에 표시될 이름과 위치 설정
plt.xticks(X + 0.50, ("A", "B", "C", "D"))

        # 막대그래프 출력
plt.show()


    # 누적 막대그래프
data = np.array([[5., 25., 50., 20.],
                [4., 23., 51., 17.],
                [6., 22., 52., 19.]])
color_list = ['b', 'g', 'r']
data_label = ["A", "B", "C"]
X = np.arange(data.shape[1])
data = np.array([[5., 5., 5., 5.], 
                 [4., 23., 51., 17.],
                 [6., 22., 52., 19.]])
for i in range(3):
    plt.bar(
        X, data[i], bottom = np.sum(data[:i], axis = 0),
        color = color_list[i], label = data_label[i]
    )
plt.legend()
plt.show()


    # 히스토그램
N = 1000
X = np.random.normal(size = N)
plt.hist(X, bins = 100)
plt.show()


    # 상자그림
data = np.random.randn(100, 5)
plt.boxplot(data)
plt.show()



# 시본
fmri = sns.load_dataset("fmri") # fmri 데이터셋 사용
sns.set_style("whitegrid") # 기본 스타일 적용
sns.lineplot(x = "timepoint", y = "signal", data = fmri) # 선 그래프 작성
plt.show()

print(fmri.sample(n = 10, random_state = 1))
sns.lineplot(x = "timepoint", y = "signal", hue = "event", data = fmri)
plt.show()


    # 회귀 그래프
tips = sns.load_dataset("tips")
sns.regplot(x = "total_bill", y = "tip", data = tips, x_ci = 95)
plt.show()

tips = sns.load_dataset("tips")
print(tips.sample(n = 10, random_state = 1))
sns.scatterplot(x = "total_bill", y = "tip", hue = "time", data = tips)
plt.show()


    # 비교 그래프
tips = sns.load_dataset("tips")
sns.countplot(x = "smoker", hue = "time", data = tips)
plt.show()


    # 막대 그래프
sns.barplot(x = "day", y = "total_bill", data = tips)
plt.show()



    # 분포를 나타내는 그래프: 바이올린 플롯과 스웜 플롯
        # 바이올린 플롯: 상자그림과 분포도를 한 번에 나타낼 수 있음
            # x축에는 범주형 데이터, y축에는 연속형 데이터
tips = sns.load_dataset("tips")
print(tips.sample(n = 10, random_state = 1))
sns.violinplot(x = "day", y = "total_bill", hue = "smoker", data = tips, palette = "muted")
plt.show()

        # 스웜 플롯: 바이올린 플롯과 같은 형태에 산점도로 데이터 분포를 나타냄
            # 매개변수 hue로 두 개 이상의 범주형 데이터를 점이 겹치지 않게 정리
            # 영역별 데이터 양을 직관적으로 보여줌
tips = sns.load_dataset("tips")
print(tips.sample(n = 10, random_state = 1))
sns.swarmplot(x = "day", y = "total_bill", hue = "smoker", data = tips, palette = "muted")
plt.show()


    # 다양한 범주형 데이터를 나타내는 패싯그리드
        # 패싯그리드: 그래프의 틀만 제공하여 적당한 그래프를 그려주는 클래스
tips = sns.load_dataset("tips")
print(tips.sample(n = 10, random_state = 1))
g = sns.FacetGrid(tips, col = "time", row = "sex")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()

g = sns.FacetGrid(tips, col = "time", row = "sex")
g.map_dataframe(sns.histplot, x = "total_bill")
plt.show()