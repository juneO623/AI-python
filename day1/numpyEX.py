np.full((3, 3), 10)

# = np.array([18, 5, 10, 2, 3, -3, -5, -30, ])

# a 원본 배열을 잘라간 b 는 부분배열. 부분배열을 수정하면 a 원본배열도 수정됨에 유의

# a = {
    
# }

# [5, 6, 7, 8] 추출
# d = a[a, :]; d

# print(d.ndim, d.shape)

# # [2, 6, 10] 추출 
# e = a[:, 1]; e

# print(e.ndim, e.shape)


# # 난수 (Random Number)
# np.random.rand(5)   # 0부터 1사이의 실수

# np.random.seed(0)   # 씨앗값
# np.random.rand(5)