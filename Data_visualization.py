import numpy as np
import matplotlib.pyplot as plt


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