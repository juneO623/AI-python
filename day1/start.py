# 숫자형
num1 = 5
num2 = 5.0
num3 = 5.0000

print(num1, num2, num3)
# 5 5.0 5.0



# 연산
num4 = 5 * 1
num5 = 5 * 1.0

print(num4, num5)
# 5 5.0



# 문자형
string1 = 'hello python world1'

string2 = "hello python world2"

string3 = '''hello python world3'''

string4 = """hello python world4"""

print(string1, string2, string3, string4, "\n")
print("Jack's favorite food is burger")
# hello python world1 hello python world2 hello python world3 hello python world4 
# Jack's favorite food is burger


print('"오늘 날씨가 참 좋다" - "아무개" ')
# "오늘 날씨가 참 좋다" - "아무개" 


print("""아우렐이우스
만물은 변화다
우리의 삶이란 우리의 생각이 변화를 만드는 (과정)이다""")

print("아우렐이우스\n만물은 변화다\n우리의 삶이란 우리의 생각이 변화를 만드는 (과정)이다")

print('''얘들아 안녕
나는 기준이라고 해''')
# 아우렐이우스
# 만물은 변화다
# 우리의 삶이란 우리의 생각이 변화를 만드는 (과정)이다

# 아우렐이우스
# 만물은 변화다
# 우리의 삶이란 우리의 생각이 변화를 만드는 (과정)이다

# 얘들아 안녕
# 나는 기준이라고 해


print("아우렐리우스 만물은 변화다 우리의 삶이란 우리의 생각이 변화를 만드는 (과정)이다")
# 아우렐리우스 만물은 변화다 우리의 삶이란 우리의 생각이 변화를 만드는 (과정)이다


# 만물은 출력해보세요
str1 = "아우렐리우스 만물은 변화다 우리의 삶이란 우리의 생각이 변화를 만드는 (과정)이다"
print(str1.split(" ")[1])

print(str1[7]+str1[8]+str1[9])
print(str1[7:10])
# 만물은
# 만물은
# 만물은




# 리스트형
a = []
b = [1, 2, 3]
c = ['wow', 'python']
d = [1, 2, 'wow']
dd = [1, 2, ['wow']]
e = [1, 2, ['wow', 'python']]
print(a, b, c, d, dd, e)
# [] [1, 2, 3] ['wow', 'python'] [1, 2, 'wow'] [1, 2, ['wow']] [1, 2, ['wow', 'python']]



a = [10, 20, 30]
print(a)

a[0] = 15
print(a)

# a[1:2] = 25
a[1:2] = ['wow', 'python', 'world']
print(a)
# [10, 20, 30]
# [15, 20, 30]
# [15, 'wow', 'python', 'world', 30]



# 딕셔너리형
game = {
    '가위' : '보',
    '보' : '가위',
    '바위' : '가위'
}
print(game['가위'])
# 보



# 튜플형
tuple1 = (1, 2, 3, 4, 5)
print(tuple1)
# (1, 2, 3, 4, 5)




list1 = [1, 2, 3, 4, 5]
print(list1)
print(type(list1))

tuple3 = tuple(list1)

print(tuple3)
print(type(tuple3))

# tuple[3] = 33
print(tuple3[3])

# [1, 2, 3, 4, 5]
# <class 'list'>
# (1, 2, 3, 4, 5)
# <class 'tuple'>
# 4




# '''
# 튜플형은 순서와 값을 모두 고정하는 형태
# 값을 벼경하려면 모두 오류가 발생함.

# # 튜플 사용하면 유용한 경우
# 1. 두 변수의 값을 맞바꿀 때
# 2. 여러 개의 값을 한꺼번에 전달할 때
# 3. 딕셔너리의 키에 값을 여러개 넣고 싶을 때
# '''



x = 5
y = 10

# x = y
# y = x

print(x, y, "\n")

temp = x
x = y
y = temp

print(x, y)

# 5 10 

# 10 5




# 튜플을 이용한 맞바꿈
print(x, y)

x, y = y, x

print(x, y)

# 10 5
# 5 10

list1 = [1, 2, 3, 4, 5]
for i in list1:
  print(i)

for i in enumerate(list1):
  # print('{} 번째 값은 {} 이다'.format(i[0], i[1]));
  print('{} 번째 값은 {} 이다'.format(*i));
# 1
# 2
# 3
# 4
# 5
# 0 번째 값은 1 이다
# 1 번째 값은 2 이다
# 2 번째 값은 3 이다
# 3 번째 값은 4 이다
# 4 번째 값은 5 이다

# for i in dict.items():
#   print('{} 점수는 {} 이다.'.format(*i))

def tuple_good():
  for i in range(100):
    return i
    
list6 = tuple_good()
print(list6)
# 0



def tuple_good2():
  return 1, 2
num1, num2 = tuple_good2()
print(num1, num2)
# 1 2


dict = {
    'x' : [1, 2, 3]
}

# # NumPy
# """

# """   