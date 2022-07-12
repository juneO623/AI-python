
# 데이터 입출력
# 외부파일 읽어오기

import pandas as pd
from pandas import Series, DataFrame

from google.colab import files
files.upload()
# 파일 22개
# auto-mpg.csv(text/csv) - 21913 bytes, last modified: 2019. 1. 10. - 100% done
# df_excelwriter.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 6108 bytes, last modified: 2021. 2. 19. - 100% done
# df_sample.csv(text/csv) - 61 bytes, last modified: 2021. 5. 4. - 100% done
# df_sample.json(application/json) - 135 bytes, last modified: 2021. 2. 19. - 100% done
# df_sample.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 5515 bytes, last modified: 2021. 2. 19. - 100% done
# df_sample_excel.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 5508 bytes, last modified: 2021. 5. 4. - 100% done
# df_sample_json.json(application/json) - 125 bytes, last modified: 2021. 5. 4. - 100% done
# df_samples_excel.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 6098 bytes, last modified: 2021. 5. 4. - 100% done
# gyonggi_population_2017.html(text/html) - 74432 bytes, last modified: 2021. 2. 22. - 100% done
# malgun.ttf(n/a) - 13457164 bytes, last modified: 2018. 9. 15. - 100% done
# read_csv_sample.csv(text/csv) - 43 bytes, last modified: 2018. 12. 26. - 100% done
# read_json_sample.json(application/json) - 472 bytes, last modified: 2018. 12. 29. - 100% done
# sample.html(text/html) - 1103 bytes, last modified: 2019. 1. 8. - 100% done
# stock price.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 10027 bytes, last modified: 2019. 3. 17. - 100% done
# stock valuation.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 10501 bytes, last modified: 2019. 1. 28. - 100% done
# stock-data.csv(text/csv) - 885 bytes, last modified: 2019. 2. 13. - 100% done
# 경기도인구데이터.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 14130 bytes, last modified: 2019. 1. 19. - 100% done
# 경기도행정구역경계.json(application/json) - 116471 bytes, last modified: 2018. 5. 13. - 100% done
# 남북한발전전력량.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 11662 bytes, last modified: 2018. 12. 27. - 100% done
# 서울지역 대학교 위치.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 7030 bytes, last modified: 2019. 1. 19. - 100% done
# 시도별 전출입 인구수.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 86316 bytes, last modified: 2019. 1. 13. - 100% done
# 주가데이터.xlsx(application/vnd.openxmlformats-officedocument.spreadsheetml.sheet) - 10793 bytes, last modified: 2019. 1. 26. - 100% done
# Saving auto-mpg.csv to auto-mpg.csv
# Saving df_excelwriter.xlsx to df_excelwriter.xlsx
# Saving df_sample.csv to df_sample.csv
# Saving df_sample.json to df_sample.json
# Saving df_sample.xlsx to df_sample.xlsx
# Saving df_sample_excel.xlsx to df_sample_excel.xlsx
# Saving df_sample_json.json to df_sample_json.json
# Saving df_samples_excel.xlsx to df_samples_excel.xlsx
# Saving gyonggi_population_2017.html to gyonggi_population_2017.html
# Saving malgun.ttf to malgun.ttf
# Saving read_csv_sample.csv to read_csv_sample.csv
# Saving read_json_sample.json to read_json_sample.json
# Saving sample.html to sample.html
# Saving stock price.xlsx to stock price.xlsx
# Saving stock valuation.xlsx to stock valuation.xlsx
# Saving stock-data.csv to stock-data.csv
# Saving 경기도인구데이터.xlsx to 경기도인구데이터.xlsx
# Saving 경기도행정구역경계.json to 경기도행정구역경계.json
# Saving 남북한발전전력량.xlsx to 남북한발전전력량.xlsx
# Saving 서울지역 대학교 위치.xlsx to 서울지역 대학교 위치.xlsx
# Saving 시도별 전출입 인구수.xlsx to 시도별 전출입 인구수.xlsx
# Saving 주가데이터.xlsx to 주가데이터.xlsx



# !ls

#  auto-mpg.csv		        read_json_sample.json
#  df_excelwriter.xlsx	        sample_data
#  df_sample.csv		        sample.html
#  df_sample_excel.xlsx	        stock-data.csv
#  df_sample.json		       'stock price.xlsx'
#  df_sample_json.json	       'stock valuation.xlsx'
#  df_samples_excel.xlsx	       '시도별 전출입 인구수.xlsx'
#  df_sample.xlsx		       '서울지역 대학교 위치.xlsx'
#  gyonggi_population_2017.html   주가데이터.xlsx
#  경기도행정구역경계.json        경기도인구데이터.xlsx
#  malgun.ttf		        남북한발전전력량.xlsx
#  read_csv_sample.csv



# csv 파일
# 파일 경로를 확인하고 변수에 저장
file_path = './read_csv_sample.csv'

# read_csv() 함수로 원데이터의 형태에 맞는 판다스 자료구조 객체ㅗ 읽어들이면 된다
df1 = pd.read_csv(file_path)
df1
#   c0	c1	c2	c3
# 0	0	1	4	7
# 1	1	2	5	8
# 2	2	3	6	9



type(df1)
# pandas.core.frame.DataFrame



df2 = pd.read_csv(file_path, index_col='c0')
df2
# 	c1	c2	c3
# c0			
# 0	1	4	7
# 1	2	5	8
# 2	3	6	9




# '''
# [참고]
# csv 파일에 따라서는 쉼표(,) 대신 공백(" "), tab("\t") 으로 텍스트를 구분하기도 함.
# 이 때, 구분자 옵션으로 delilm= , sep= 에 알맞게 입력하면 된다.
# '''




# excel 파일 읽어들이기
# read_excel()
df1 = pd.read_excel('./남북한발전전력량.xlsx')
# df2 = pd.read_excel('./남북한발전전력량.xlsx', )
# 	전력량 (억㎾h)	발전 전력별	1990	1991	1992	1993	1994	1995	1996	1997	...	2007	2008	2009	2010	2011	2012	2013	2014	2015	2016
# 0	남한	합계	1077	1186	1310	1444	1650	1847	2055	2244	...	4031	4224	4336	4747	4969	5096	5171	5220	5281	5404
# 1	NaN	수력	64	51	49	60	41	55	52	54	...	50	56	56	65	78	77	84	78	58	66
# 2	NaN	화력	484	573	696	803	1022	1122	1264	1420	...	2551	2658	2802	3196	3343	3430	3581	3427	3402	3523
# 3	NaN	원자력	529	563	565	581	587	670	739	771	...	1429	1510	1478	1486	1547	1503	1388	1564	1648	1620
# 4	NaN	신재생	-	-	-	-	-	-	-	-	...	-	-	-	-	-	86	118	151	173	195
# 5	북한	합계	277	263	247	221	231	230	213	193	...	236	255	235	237	211	215	221	216	190	239
# 6	NaN	수력	156	150	142	133	138	142	125	107	...	133	141	125	134	132	135	139	130	100	128
# 7	NaN	화력	121	113	105	88	93	88	88	86	...	103	114	110	103	79	80	82	86	90	111
# 8	NaN	원자력	-	-	-	-	-	-	-	-	...	-	-	-	-	-	-	-	-	-	-




df1
# 전력량 (억㎾h)	발전 전력별	1990	1991	1992	1993	1994	1995	1996	1997	...	2007	2008	2009	2010	2011	2012	2013	2014	2015	2016
# 0	남한	합계	1077	1186	1310	1444	1650	1847	2055	2244	...	4031	4224	4336	4747	4969	5096	5171	5220	5281	5404
# 1	NaN	수력	64	51	49	60	41	55	52	54	...	50	56	56	65	78	77	84	78	58	66
# 2	NaN	화력	484	573	696	803	1022	1122	1264	1420	...	2551	2658	2802	3196	3343	3430	3581	3427	3402	3523
# 3	NaN	원자력	529	563	565	581	587	670	739	771	...	1429	1510	1478	1486	1547	1503	1388	1564	1648	1620
# 4	NaN	신재생	-	-	-	-	-	-	-	-	...	-	-	-	-	-	86	118	151	173	195
# 5	북한	합계	277	263	247	221	231	230	213	193	...	236	255	235	237	211	215	221	216	190	239
# 6	NaN	수력	156	150	142	133	138	142	125	107	...	133	141	125	134	132	135	139	130	100	128
# 7	NaN	화력	121	113	105	88	93	88	88	86	...	103	114	110	103	79	80	82	86	90	111
# 8	NaN	원자력	-	-	-	-	-	-	-	-	...	-	-	-	-	-	-	-	-	-	-



# df2

type(df1)
# pandas.core.frame.DataFrame



# xlrd, openpyxl
# !pip install openpyxl

# JSON 파일 읽어들이기

# web 에서 읽어들이기
url = 'sample.html'


# HTML 웹페이지 표(table) 를 읽어들이기 가능
tables = pd.read_html(url)
tables
# [   Unnamed: 0  c0  c1  c2  c3
#  0           0   0   1   4   7
#  1           1   1   2   5   8
#  2           2   2   3   6   9,
#           name  year        developer  opensource
#  0       NumPy  2006  Travis Oliphant        True
#  1  matplotlib  2003   John D. Hunter        True
#  2      pandas  2008    Wes Mckinneye        True]



for i in range(len(tables)):
  print('tables[%s]' % i)
  print(tables[i])
  print('\n')
# tables[0]
#    Unnamed: 0  c0  c1  c2  c3
# 0           0   0   1   4   7
# 1           1   1   2   5   8
# 2           2   2   3   6   9


# tables[1]
#          name  year        developer  opensource
# 0       NumPy  2006  Travis Oliphant        True
# 1  matplotlib  2003   John D. Hunter        True
# 2      pandas  2008    Wes Mckinneye        True




# 각 테이블을 데이터프레임 객체로 만들어보세요
df1 = DataFrame(tables[0])
df2 = DataFrame(tables[1])



df1
# Unnamed: 0	c0	c1	c2	c3
# 0	            0	0	1	4	7
# 1	            1	1	2	5	8
# 2	            2	2	3	6	9


df2
#   name	year	developer	opensource
# 0	NumPy	2006	Travis Oliphant	True
# 1	matplotlib	2003	John D. Hunter	True
# 2	pandas	2008	Wes Mckinneye	True



# csv 저장하기
data = {
    'name' : ['Jerry', 'Riah', 'Paul'],
    'algo' : ['A', 'A', 'B'],
    'basic' : ['B', 'B+', 'B'],
    'python' : ['B+', 'C', 'C+']
}
data
# {'algo': ['A', 'A', 'B'],
#  'basic': ['B', 'B+', 'B'],
#  'name': ['Jerry', 'Riah', 'Paul'],
#  'python': ['B+', 'C', 'C+']}



# df.to_csv('df_sample.csv')

files.download('df_sample.csv')

# data 를 JSON으로 저장, excel로 저장해보세요
#   df_sample.json         df_sample.xlsx
df = DataFrame(data)
df.set_index('name', inplace=True)

files.download('df_sample.json')

files.download('df_sample.xlsx')


# 여러개의 데이터 플임을 엑셀파일 하나에 저장
data1 = {
    'name' : ['Jerry', 'Riah', 'Paul'],
    'algo' : ['A', 'A', 'B'],
    'basic' : ['B', 'B+', 'B'],
    'python' : ['B+', 'C', 'C+']
}
data2 = {
    'c0' : [1, 2, 3],
    'c1' : [4, 5, 6],
    'c2' : [7, 8, 9],
    'c3' : [10, 11, 12],
    'c4' : [13, 14 ,15]
}



df1 = DataFrame(data1)
df1.set_index('name', inplace=True)

df2 = DataFrame(data2)
df2.set_index('c0', inplace=True)


# df1은 sheet1로, df2는 sheet2로 저장
# ExcelWriter 와 함께 사용

writer = pd.ExcelWriter('./df_excelWriter.xlsx')
df1.to_excel(writer, sheet_name='sheet1')
df2.to_excel(writer, sheet_name='sheet2')
writer.save()

files.download('./df_excelWriter.xlsx')



# 데이터 살펴보기


df = pd.read_csv('./auto-mpg.csv')
df
# 	18.0	8	307.0	130.0	3504.	12.0	70	1	chevrolet chevelle malibu
# 0	15.0	8	350.0	165.0	3693.0	11.5	70	1	buick skylark 320
# 1	18.0	8	318.0	150.0	3436.0	11.0	70	1	plymouth satellite
# 2	16.0	8	304.0	150.0	3433.0	12.0	70	1	amc rebel sst
# 3	17.0	8	302.0	140.0	3449.0	10.5	70	1	ford torino
# 4	15.0	8	429.0	198.0	4341.0	10.0	70	1	ford galaxie 500
# ...	...	...	...	...	...	...	...	...	...
# 392	27.0	4	140.0	86.00	2790.0	15.6	82	1	ford mustang gl
# 393	44.0	4	97.0	52.00	2130.0	24.6	82	2	vw pickup
# 394	32.0	4	135.0	84.00	2295.0	11.6	82	1	dodge rampage
# 395	28.0	4	120.0	79.00	2625.0	18.6	82	1	ford ranger
# 396	31.0	4	119.0	82.00	2720.0	19.4	82	1	chevy s-10
# 397 rows × 9 columns



# '''
# 1. mpg                        연비                      : continuous
# 2. cylinders                  실린더수                  : multi-valued discrete
# 3. displacement               배기량                    : continuous
# 4. horsepower                 마력                      : continuous
# 5. weight                     차량무게                  : continuous
# 6. acceleration               가속능력                  : continuous
# 7. model year                 출시년도                  : multi-valued discrete
# 8. origin                     제조국                    : multi-valued discrete
# 9. car name                   자동차이름                : string (unique for each instance)
# '''



# 데이터프레임의 데이터 일부만 확인
df. head()
# 	18.0	8	307.0	130.0	3504.	12.0	70	1	chevrolet chevelle malibu
# 0	15.0	8	350.0	165.0	3693.0	11.5	70	1	buick skylark 320
# 1	18.0	8	318.0	150.0	3436.0	11.0	70	1	plymouth satellite
# 2	16.0	8	304.0	150.0	3433.0	12.0	70	1	amc rebel sst
# 3	17.0	8	302.0	140.0	3449.0	10.5	70	1	ford torino
# 4	15.0	8	429.0	198.0	4341.0	10.0	70	1	ford galaxie 500


df.head(10)  # default 5
# 	18.0	8	307.0	130.0	3504.	12.0	70	1	chevrolet chevelle malibu
# 0	15.0	8	350.0	165.0	3693.0	11.5	70	1	buick skylark 320
# 1	18.0	8	318.0	150.0	3436.0	11.0	70	1	plymouth satellite
# 2	16.0	8	304.0	150.0	3433.0	12.0	70	1	amc rebel sst
# 3	17.0	8	302.0	140.0	3449.0	10.5	70	1	ford torino
# 4	15.0	8	429.0	198.0	4341.0	10.0	70	1	ford galaxie 500
# 5	14.0	8	454.0	220.0	4354.0	9.0	70	1	chevrolet impala
# 6	14.0	8	440.0	215.0	4312.0	8.5	70	1	plymouth fury iii
# 7	14.0	8	455.0	225.0	4425.0	10.0	70	1	pontiac catalina
# 8	15.0	8	390.0	190.0	3850.0	8.5	70	1	amc ambassador dpl
# 9	15.0	8	383.0	170.0	3563.0	10.0	70	1	dodge challenger se



df.tail()
# 	    18.0	8	307.0	130.0	3504.	12.0	70	1	chevrolet chevelle malibu
# 392	27.0	4	140.0	86.00	2790.0	15.6	82	1	ford mustang gl
# 393	44.0	4	97.0	52.00	2130.0	24.6	82	2	vw pickup
# 394	32.0	4	135.0	84.00	2295.0	11.6	82	1	dodge rampage
# 395	28.0	4	120.0	79.00	2625.0	18.6	82	1	ford ranger
# 396	31.0	4	119.0	82.00	2720.0	19.4	82	1	chevy s-10


# 열 이름을 지정해서 데이터세트 구성
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model year', 'origin', 'car name']
df.head()
#   mpg	cylinders	displacement	horsepower	weight	acceleration	model year	origin	car name
# 0	15.0	8	350.0	165.0	3693.0	11.5	70	1	buick skylark 320
# 1	18.0	8	318.0	150.0	3436.0	11.0	70	1	plymouth satellite
# 2	16.0	8	304.0	150.0	3433.0	12.0	70	1	amc rebel sst
# 3	17.0	8	302.0	140.0	3449.0	10.5	70	1	ford torino
# 4	15.0	8	429.0	198.0	4341.0	10.0	70	1	ford galaxie 500


df.mpg.dtypes
# dtype('float64')


df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 15 columns):
#  #   Column       Non-Null Count  Dtype   
# ---  ------       --------------  -----   
#  0   survived     891 non-null    int64   
#  1   pclass       891 non-null    int64   
#  2   sex          891 non-null    object  
#  3   age          714 non-null    float64 
#  4   sibsp        891 non-null    int64   
#  5   parch        891 non-null    int64   
#  6   fare         891 non-null    float64 
#  7   embarked     889 non-null    object  
#  8   class        891 non-null    category
#  9   who          891 non-null    object  
#  10  adult_male   891 non-null    bool    
#  11  deck         203 non-null    category
#  12  embark_town  889 non-null    object  
#  13  alive        891 non-null    object  
#  14  alone        891 non-null    bool    
# dtypes: bool(2), category(2), float64(2), int64(4), object(5)
# memory usage: 80.7+ KB



# 데이터 개수 확인
df.count()
# mpg             397
# cylinders       397
# displacement    397
# horsepower      397
# weight          397
# acceleration    397
# model year      397
# origin          397
# car name        397
# dtype: int64



# 평균
df.mean()
# /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
  
# mpg               23.528463
# cylinders          5.448363
# displacement     193.139798
# weight          2969.080605
# acceleration      15.577078
# model year        76.025189
# origin             1.574307
# dtype: float64



# 특정 열 평균값
df.mpg.mean()
# 23.528463476070527



# 특정 열 평균값을 - 열 두 개 선택
# mpg, weight 열들의 평균 각각 구하기
# print(df.mpg.mean(), df.weight.mean())  # 
df[['mpg', 'weight']].mean()
# mpg         23.528463
# weight    2969.080605
# dtype: float64



df.corr()   # 상관계수 (숫자가 높을 수록 영향을 준다)
#       cylinders	displacement	weight	acceleration	model   year	    origin
# mpg	1.000000	-0.775412	-0.803972	-0.831558	0.419133	0.578667	0.562894
# cylinders	-0.775412	1.000000	0.950718	0.896623	-0.503016	-0.344729	-0.561796
# displacement	-0.803972	0.950718	1.000000	0.932957	-0.542083	-0.367470	-0.608749
# weight	-0.831558	0.896623	0.932957	1.000000	-0.416488	-0.305150	-0.580552
# acceleration	0.419133	-0.503016	-0.542083	-0.416488	1.000000	0.284376	0.204102
# model year	0.578667	-0.344729	-0.367470	-0.305150	0.284376	1.000000	0.178441
# origin	0.562894	-0.561796	-0.608749	-0.580552	0.204102	0.178441	1.000000



df[['mpg', 'weight']].corr()
#       mpg	        weight
# mpg	1.000000	-0.831558
# weight	-0.831558	1.000000


# 중간값, 최대값, 최소값, 표준편차
# median(), max(), min(), std()

df.describe()
#         mpg	        cylinders	displacement	weight	acceleration	model year	origin
# count	397.000000	397.000000	397.000000	397.000000	397.000000	397.000000	397.000000
# mean	23.528463	5.448363	193.139798	2969.080605	15.577078	76.025189	1.574307
# std	7.820926	1.698329	104.244898	847.485218	2.755326	3.689922	0.802549
# min	9.000000	3.000000	68.000000	1613.000000	8.000000	70.000000	1.000000
# 25%	17.500000	4.000000	104.000000	2223.000000	13.900000	73.000000	1.000000
# 50%	23.000000	4.000000	146.000000	2800.000000	15.500000	76.000000	1.000000
# 75%	29.000000	8.000000	262.000000	3609.000000	17.200000	79.000000	2.000000
# max	46.600000	8.000000	455.000000	5140.000000	24.800000	82.000000	3.000000



df.describe(include='all')
#       mpg	    cylinders	displacement	horsepower	weight	acceleration	model year	origin	car name
# count	397.000000	397.000000	397.000000	397	397.000000	397.000000	397.000000	397.000000	397
# unique	NaN	NaN	NaN	94	NaN	NaN	NaN	NaN	305
# top	NaN	NaN	NaN	150.0	NaN	NaN	NaN	NaN	ford pinto
# freq	NaN	NaN	NaN	22	NaN	NaN	NaN	NaN	6
# mean	23.528463	5.448363	193.139798	NaN	2969.080605	15.577078	76.025189	1.574307	NaN
# std	7.820926	1.698329	104.244898	NaN	847.485218	2.755326	3.689922	0.802549	NaN
# min	9.000000	3.000000	68.000000	NaN	1613.000000	8.000000	70.000000	1.000000	NaN
# 25%	17.500000	4.000000	104.000000	NaN	2223.000000	13.900000	73.000000	1.000000	NaN
# 50%	23.000000	4.000000	146.000000	NaN	2800.000000	15.500000	76.000000	1.000000	NaN
# 75%	29.000000	8.000000	262.000000	NaN	3609.000000	17.200000	79.000000	2.000000	NaN
# max	46.600000	8.000000	455.000000	NaN	5140.000000	24.800000	82.000000	3.000000	NaN



# 그래프 도구를 이용한 탐색

df = pd.read_excel('남북한발전전력량.xlsx')
df.head()
#   전력량 (억㎾h)	발전 전력별	1990	1991	1992	1993	1994	1995	1996	1997	...	2007	2008	2009	2010	2011	2012	2013	2014	2015	2016
# 0	남한	합계	1077	1186	1310	1444	1650	1847	2055	2244	...	4031	4224	4336	4747	4969	5096	5171	5220	5281	5404
# 1	NaN	수력	64	51	49	60	41	55	52	54	...	50	56	56	65	78	77	84	78	58	66
# 2	NaN	화력	484	573	696	803	1022	1122	1264	1420	...	2551	2658	2802	3196	3343	3430	3581	3427	3402	3523
# 3	NaN	원자력	529	563	565	581	587	670	739	771	...	1429	1510	1478	1486	1547	1503	1388	1564	1648	1620
# 4	NaN	신재생	-	-	-	-	-	-	-	-	...	-	-	-	-	-	86	118	151	173	195
# 5 rows × 29 columns



# 남한, 북한 합계 정보
# df.loc[0]
df_sn = df.iloc[[0, 5], 2:]
df_sn
# 	1990	1991	1992	1993	1994	1995	1996	1997	1998	1999	...	2007	2008	2009	2010	2011	2012	2013	2014	2015	2016
# 0	1077	1186	1310	1444	1650	1847	2055	2244	2153	2393	...	4031	4224	4336	4747	4969	5096	5171	5220	5281	5404
# 5	277	263	247	221	231	230	213	193	170	186	...	236	255	235	237	211	215	221	216	190	239
# 2 rows × 27 columns



df_sn.index = ['South', 'North']
df_sn.columns = df_sn.columns.map(int)    # 열 자료형을 정수형으로 변경
df_sn.head()
#       1990	1991	1992	1993	1994	1995	1996	1997	1998	1999	...	2007	2008	2009	2010	2011	2012	2013	2014	2015	2016
# South	1077	1186	1310	1444	1650	1847	2055	2244	2153	2393	...	4031	4224	4336	4747	4969	5096	5171	5220	5281	5404
# North	277	263	247	221	231	230	213	193	170	186	...	236	255	235	237	211	215	221	216	190	239
# 2 rows × 27 columns



df_sn.plot()    # 그래프가 그려짐
# <matplotlib.axes._subplots.AxesSubplot at 0x7f0ebd3c1b10>


# 행, 열 전치
tdf_sn = df_sn.T
tdf_sn.plot()     # 행 열 바꾸고 그래프로 표현
# 남한은 점점 높아짐, 북한은 쭉 유지
# <matplotlib.axes._subplots.AxesSubplot at 0x7f0ebccd6750>


tdf_sn.head()
#       South	North
# 1990	1077	277
# 1991	1186	263
# 1992	1310	247
# 1993	1444	221
# 1994	1650	231



# 막대 그래프
tdf_sn.plot(kind='bar')   # 데이터를 막대 그래프로 표현
# <matplotlib.axes._subplots.AxesSubplot at 0x7f0ebcc29fd0>



# 히스토그램
tdf_sn.plot(kind='hist')
# <matplotlib.axes._subplots.AxesSubplot at 0x7f0ebc947410>



# 산점도
df = pd.read_csv('./auto-mpg.csv', header=None)
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'model year', 'origin', 'car name']
df.head()
#   mpg	cylinders	displacement	horsepower	weight	acceleration	model year	origin	car name
# 0	18.0	8	307.0	130.0	3504.0	12.0	70	1	chevrolet chevelle malibu
# 1	15.0	8	350.0	165.0	3693.0	11.5	70	1	buick skylark 320
# 2	18.0	8	318.0	150.0	3436.0	11.0	70	1	plymouth satellite
# 3	16.0	8	304.0	150.0	3433.0	12.0	70	1	amc rebel sst
# 4	17.0	8	302.0	140.0	3449.0	10.5	70	1	ford torino



df.plot(x='weight', y='mpg', kind='scatter')  # x축에는 무게 (독립변수), y축은 연비 (종속변수)
# <matplotlib.axes._subplots.AxesSubplot at 0x7f0ebc9b6510>



# 차의 무게가 높아질 수록 연비가 전반적으로 낮아지는 경향이 보인다
# 역상관관계를 갖는다 (오른쪽으로 커질수록 낮아지는 그래프)

# 박스플롯 (상자 수염 그림)
df.mpg.plot(kind='box')
# <matplotlib.axes._subplots.AxesSubplot at 0x7f0ebca8e9d0>
# 가운데 초록선이 평균


# 박스플롯2
df[['mpg', 'cylinders']].plot(kind='box')
# <matplotlib.axes._subplots.AxesSubplot at 0x7f0ebc7d4ed0>





# 데이터 (사)전처리
# 누락데이터 (NaN : Not a Number) 처리

import seaborn as sns

# titanic
df = sns.load_dataset('titanic')
df.head()
# 	survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone
# 0	0	3	male	22.0	1	0	7.2500	S	Third	man	True	NaN	Southampton	no	False
# 1	1	1	female	38.0	1	0	71.2833	C	First	woman	False	C	Cherbourg	yes	False
# 2	1	3	female	26.0	0	0	7.9250	S	Third	woman	False	NaN	Southampton	yes	True
# 3	1	1	female	35.0	1	0	53.1000	S	First	woman	False	C	Southampton	yes	False
# 4	0	3	male	35.0	0	0	8.0500	S	Third	man	True	NaN	Southampton	no	True


# 기본 정보 확인
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 15 columns):
#  #   Column       Non-Null Count  Dtype   
# ---  ------       --------------  -----   
#  0   survived     891 non-null    int64   
#  1   pclass       891 non-null    int64   
#  2   sex          891 non-null    object  
#  3   age          714 non-null    float64 
#  4   sibsp        891 non-null    int64   
#  5   parch        891 non-null    int64   
#  6   fare         891 non-null    float64 
#  7   embarked     889 non-null    object  
#  8   class        891 non-null    category
#  9   who          891 non-null    object  
#  10  adult_male   891 non-null    bool    
#  11  deck         203 non-null    category
#  12  embark_town  889 non-null    object  
#  13  alive        891 non-null    object  
#  14  alone        891 non-null    bool    
# dtypes: bool(2), category(2), float64(2), int64(4), object(5)
# memory usage: 80.7+ KB



# deck 열의 NaN 개수 확인
# df.deck.value_counts()
# C    59
# B    47
# D    33
# E    32
# A    15
# F    13
# G     4
# Name: deck, dtype: int64


# deck 열의 NaN 개수 확인2
# df.deck.value_counts(dropna=False)
# NaN    688
# C       59
# B       47
# D       33
# E       32
# A       15
# F       13
# G        4
# Name: deck, dtype: int64



# null인지 아닌지 판별
df.deck.isnull()
# 0       True
# 1      False
# 2       True
# 3      False
# 4       True
#        ...  
# 886     True
# 887    False
# 888     True
# 889    False
# 890     True
# Name: deck, Length: 891, dtype: bool


# null이면 True(1) 아니면 False(0)으로 변환되어서 총합
df.deck.isnull().sum()
# 688


# 누락 데이터 제거
df_tresh = df.dropna(thresh=500, axis=1)
df_tresh.columns
# Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
#        'embarked', 'class', 'who', 'adult_male', 'embark_town', 'alive',
#        'alone'],
#       dtype='object')



df = pd.read_csv('./auto-mpg.csv', header =None)
df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin', 'car name']
df.head()
# 	mpg	cylinders	displacement	horsepower	weight	acceleration	model year	origin	car name
# 0	18.0	8	307.0	130.0	3504.0	12.0	70	1	chevrolet chevelle malibu
# 1	15.0	8	350.0	165.0	3693.0	11.5	70	1	buick skylark 320
# 2	18.0	8	318.0	150.0	3436.0	11.0	70	1	plymouth satellite
# 3	16.0	8	304.0	150.0	3433.0	12.0	70	1	amc rebel sst
# 4	17.0	8	302.0	140.0	3449.0	10.5	70	1	ford torino



df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 398 entries, 0 to 397
# Data columns (total 9 columns):
#  #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   mpg           398 non-null    float64
#  1   cylinders     398 non-null    int64  
#  2   displacement  398 non-null    float64
#  3   horsepower    398 non-null    object 
#  4   weight        398 non-null    float64
#  5   acceleration  398 non-null    float64
#  6   model year    398 non-null    int64  
#  7   origin        398 non-null    int64  
#  8   car name      398 non-null    object 
# dtypes: float64(4), int64(3), object(2)
# memory usage: 28.1+ KB



df.horsepower.unique()
# array(['130.0', '165.0', '150.0', '140.0', '198.0', '220.0', '215.0',
#        '225.0', '190.0', '170.0', '160.0', '95.00', '97.00', '85.00',
#        '88.00', '46.00', '87.00', '90.00', '113.0', '200.0', '210.0',
#        '193.0', '?', '100.0', '105.0', '175.0', '153.0', '180.0', '110.0',
#        '72.00', '86.00', '70.00', '76.00', '65.00', '69.00', '60.00',
#        '80.00', '54.00', '208.0', '155.0', '112.0', '92.00', '145.0',
#        '137.0', '158.0', '167.0', '94.00', '107.0', '230.0', '49.00',
#        '75.00', '91.00', '122.0', '67.00', '83.00', '78.00', '52.00',
#        '61.00', '93.00', '148.0', '129.0', '96.00', '71.00', '98.00',
#        '115.0', '53.00', '81.00', '79.00', '120.0', '152.0', '102.0',
#        '108.0', '68.00', '58.00', '149.0', '89.00', '63.00', '48.00',
#        '66.00', '139.0', '103.0', '125.0', '133.0', '138.0', '135.0',
#        '142.0', '77.00', '62.00', '132.0', '84.00', '64.00', '74.00',
#        '116.0', '82.00'], dtype=object)



# '?' -> 누락데이터로 변경 -> 누락데이터 제거 -> 형변환
import numpy as np
df.horsepower.replace('?', np.nan, inplace=True)


df.horsepower.unique()
# array(['130.0', '165.0', '150.0', '140.0', '198.0', '220.0', '215.0',
#        '225.0', '190.0', '170.0', '160.0', '95.00', '97.00', '85.00',
#        '88.00', '46.00', '87.00', '90.00', '113.0', '200.0', '210.0',
#        '193.0', nan, '100.0', '105.0', '175.0', '153.0', '180.0', '110.0',
#        '72.00', '86.00', '70.00', '76.00', '65.00', '69.00', '60.00',
#        '80.00', '54.00', '208.0', '155.0', '112.0', '92.00', '145.0',
#        '137.0', '158.0', '167.0', '94.00', '107.0', '230.0', '49.00',
#        '75.00', '91.00', '122.0', '67.00', '83.00', '78.00', '52.00',
#        '61.00', '93.00', '148.0', '129.0', '96.00', '71.00', '98.00',
#        '115.0', '53.00', '81.00', '79.00', '120.0', '152.0', '102.0',
#        '108.0', '68.00', '58.00', '149.0', '89.00', '63.00', '48.00',
#        '66.00', '139.0', '103.0', '125.0', '133.0', '138.0', '135.0',
#        '142.0', '77.00', '62.00', '132.0', '84.00', '64.00', '74.00',
#        '116.0', '82.00'], dtype=object)



df.dropna(subset=['horsepower'], axis=0, inplace=True)
df.horsepower.unique()
# array(['130.0', '165.0', '150.0', '140.0', '198.0', '220.0', '215.0',
#        '225.0', '190.0', '170.0', '160.0', '95.00', '97.00', '85.00',
#        '88.00', '46.00', '87.00', '90.00', '113.0', '200.0', '210.0',
#        '193.0', '100.0', '105.0', '175.0', '153.0', '180.0', '110.0',
#        '72.00', '86.00', '70.00', '76.00', '65.00', '69.00', '60.00',
#        '80.00', '54.00', '208.0', '155.0', '112.0', '92.00', '145.0',
#        '137.0', '158.0', '167.0', '94.00', '107.0', '230.0', '49.00',
#        '75.00', '91.00', '122.0', '67.00', '83.00', '78.00', '52.00',
#        '61.00', '93.00', '148.0', '129.0', '96.00', '71.00', '98.00',
#        '115.0', '53.00', '81.00', '79.00', '120.0', '152.0', '102.0',
#        '108.0', '68.00', '58.00', '149.0', '89.00', '63.00', '48.00',
#        '66.00', '139.0', '103.0', '125.0', '133.0', '138.0', '135.0',
#        '142.0', '77.00', '62.00', '132.0', '84.00', '64.00', '74.00',
#        '116.0', '82.00'], dtype=object)
