# Pandas
import pandas as pd
from pandas import Series, DataFrame

# Series 객체
# 일차원 배열 같은 자료구조 객체


obj = Series([3, 22, 34, 11])
obj
# 0     3
# 1    22
# 2    34
# 3    11
# dtype: int64


print(obj.values)
print("------------------")
print(obj.index)
# [ 3 22 34 11]
# ------------------
# RangeIndex(start=0, stop=4, step=1)


print([1, 2, 3, 4])
# [1, 2, 3, 4]


# index 가 보이므로 지정 가능
obj2 = Series([4, 5, 6, 2], index = ['c', 'd', 'e', 'f'])
obj2
# c    4
# d    5
# e    6
# f    2
# dtype: int64


# indexing
obj2['c']
# 4


# 각 요소를 한꺼번에 지정
# c, d, f 출력해보세요
obj2[['c', 'd', 'f']]
# c    4
# d    5
# f    2
# dtype: int64


# 각 요소별 연산
obj * 2
# 0     6
# 1    44
# 2    68
# 3    22
# dtype: int64



# 딕셔너리와 거의 유사하므로 대체 가능
data = {
    'kim' : 3000,
    'hong' : 2000,
    'kang' : 1000,
    'lee' : 2400
}

obj3 = Series(data)
# kim     3000
# hong    2000
# kang    1000
# lee     2400
# dtype: int64




name = ['woo', 'hong', 'kang', 'lee']
obj4 = Series(data, index=name)
obj4
# woo        NaN
# hong    2000.0
# kang    1000.0
# lee     2400.0
# dtype: float64



# 누락 데이터 찾는 함수 : isnull, notnull
print(pd.isnull(obj4))
print(pd.notnull(obj4))
# woo      True
# hong    False
# kang    False
# lee     False
# dtype: bool
# woo     False
# hong     True
# kang     True
# lee      True
# dtype: bool



obj3
# kim     3000
# hong    2000
# kang    1000
# lee     2400
# dtype: int64



# Series 객체 이름, Series 색인 객체의 이름  모두 name 속성이 있음
obj3.name = '최고득점'
obj3
# kim     3000
# hong    2000
# kang    1000
# lee     2400
# Name: 최고득점, dtype: int64



obj3.index.name = '이름'
obj3
# 이름
# kim     3000
# hong    2000
# kang    1000
# lee     2400
# Name: 최고득점, dtype: int64



# DataFrame 자료구조 객체
# 2차원리스트(배열)과 같은 자료구조 객체


x = DataFrame([
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
])

x
#   0	1	2
# 0	1	2	3
# 1	4	5	6
# 2	7	8	9


# 딕셔너리로 데이터 프레임 대체 가능
data = {
    'city' : ['서울', '부산', '광주', '대구'],
    'year' : [2000, 2001, 2002, 2002],
    'pop' : [4000, 2000, 1000, 1000]
}
df = DataFrame(data)
df


#    city	year	pop
# 0	서울	2000	4000
# 1	부산	2001	2000
# 2	광주	2002	1000
# 3	대구	2002	1000




# 컬럼 순서 변경
df = DataFrame(data, columns=['year', 'city', 'pop'])
df

#     year	city	pop
# 0	2000	서울	4000
# 1	2001	부산	2000
# 2	2002	광주	1000
# 3	2002	대구	1000



# 인덱스 지정
df2 = DataFrame(data, columns=['year', 'city', 'pop', 'debt'],
               index=['one', 'two', 'three', 'four'])
df2
#       year	city	pop	    debt
# one	2000	서울	4000	NaN
# two	2001	부산	2000	NaN
# three	2002	광주	1000	NaN
# four	2002	대구	1000	NaN



# 인덱싱
df2['city']

# Series로 리턴 됨

# one      서울
# two      부산
# three    광주
# four     대구
# Name: city, dtype: object




print(df2.columns, df2.index)
# Index(['year', 'city', 'pop', 'debt'], dtype='object') Index(['one', 'two', 'three', 'four'], dtype='object')



# 행 단위로 추출
df2.loc['three']
# year    2002
# city      광주
# pop     1000
# debt     NaN
# Name: three, dtype: object




# 값 삽입
# df2.debt
df2['debt'] = 1000
df2
#       year	city	pop	    debt
# one	2000	서울	4000	1000
# two	2001	부산	2000	1000
# three	2002	광주	1000	1000
# four	2002	대구	1000	1000



# Series 이용해서 값 삽입 (유의 - 인덱스 매칭이 필요)
val = Series([1000, 2000, 3000, 4000], index=['one', 'two', 'three', 'four'])

df2['debt'] = val
df2
# 	    year	city	pop	    debt
# one	2000	서울	4000	1000
# two	2001	부산	2000	2000
# three	2002	광주	1000	3000
# four	2002	대구	1000	4000




# 값 삽입 - 연산의 결과 T/F 를 삽입
df2['cap'] = df2.city == '서울'
df2
#       year	city	pop	    debt	cap
# one	2000	서울	4000	1000	True
# two	2001	부산	2000	2000	False
# three	2002	광주	1000	3000	False
# four	2002	대구	1000	4000	False



data2 = {
    'seoul' : {2019 : 20, 2020 : 30},
    'busan' : {2018 : 10, 2019 : 200, 2020 : 300}
}

df3 = DataFrame(data2)
df3
#    	seoul	busan
# 2019	20.0	200
# 2020	30.0	300
# 2018	NaN	10



# 전치행렬 (행과 열 바꾸기)
df3.T
# 	    2019	2020	2018
# seoul	20.0	30.0	NaN
# busan	200.0	300.0	10.0



# 데이터만 추출
df3.values
# array([[ 20., 200.],
#        [ 30., 300.],
#        [ nan,  10.]])