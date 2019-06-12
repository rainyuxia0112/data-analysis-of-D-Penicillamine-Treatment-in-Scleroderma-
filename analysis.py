################################ 导入所有数据
import pandas as pd
def load_large_dta(fname):
    import sys

    reader = pd.read_stata(fname, iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(100*1000)
        while len(chunk) > 0:
            df = df.append(chunk, ignore_index=True)
            chunk = reader.get_chunk(100*1000)
            print ('.')
            sys.stdout.flush()
    except (StopIteration, KeyboardInterrupt):
        pass

    print ('\nloaded {} rows'.format(len(df)))

    return df


origin = load_large_dta('/Users/rain/Desktop/200c/200cTermProject.dta')
origin.visitdat = pd.to_datetime(origin['visitdat'], errors='coerce')
data = origin
# 先排序（按照group）
data  = origin.groupby(by = 'patid').apply(lambda x: x.sort_values('visitdat'))
data.reset_index(drop=True, inplace=True)

data  = data.dropna(subset=['visitdat', 'haq'])


############################ 分类排序建新col

def minus(df):
    l = []
    for i in range(len(df)):
        l.append(df.iloc[i, 2] - df.iloc[0, 2])
    df['times'] = l    
    return (df)        

data = data.groupby('patid').apply(minus) #变成了很多个小的dataframe

########################### 计算次数！！ （可以直接在minus function里面做（直接
# 转换为第几次
def check(time):
    if time == pd.Timedelta(0,'D'):
        times = 1
    elif  pd.Timedelta(120, 'D')< time < pd.Timedelta(240, 'D'):
        times = 2
    elif  pd.Timedelta(300, 'D')<time< pd.Timedelta(420, 'D'):
        times = 3
    elif pd.Timedelta(480, 'D') < time < pd.Timedelta(600, 'D'):
        times = 4
    elif pd.Timedelta(660, 'D') < time:
        times = 5
    else:
        times = 0
    return (times)
data['times'] = list(map(check, data['times']))

# check their data type
data.info()

#########################  
info = data.groupby('patid').min()
import matplotlib.pyplot as plt
# 画race 的图
race = ['white', 'Black', 'Asian', 'other', 'Native Hawaiian']
info.race.value_counts().plot.bar(rot = 0)
plt.xticks(range(5), race)
plt.xlabel('race')
plt.ylabel('count_number')
# 画 sex的图
sex = ['female', 'male', 'strange data']
info.sex.value_counts().plot.bar(rot = 0)
plt.xticks(range(3), sex)
plt.ylabel('count_number')
plt.xlabel('gender')
plt.title('gender distribution')
# 画 age的hist
info.age.plot(kind = 'hist')
plt.xlabel('age')
plt.ylabel('count_number')
plt.title('age distribution')

# 画 sbp的hist
info.sbp.plot(kind = 'hist')
plt.xlabel('sbp')
plt.ylabel('count_number')
plt.title('sbp distribution')

# 画 cpk的hist
info.cpk.plot(kind = 'hist')
plt.xlabel('cpk')
plt.ylabel('count_number')
plt.title('cpk distribution')

# 画 haq 的hist
info.haq.plot(kind = 'hist')
plt.xlabel('haq')
plt.ylabel('count_number')
plt.title('haq distribution')

# 画 cardrate 的hist
info.cardrate .plot(kind = 'hist')
plt.xlabel('cardrate')
plt.ylabel('count_number')
plt.title('cardrate distribution')


# 每次有多少人出现（ 看 times） 多少人完成5次
data.times.value_counts()
len(info[(data.groupby('patid')['times']).count() == 5])
# plot bar  plot
data.times.value_counts().plot.bar(rot = 0)
plt.title('the number of visit each time')

################3 完成者与非完成者的不同
complete = list(info[(data.groupby('patid')['times']).count() == 5].index) # 67 个人
null_complete = list(info[(data.groupby('patid')['times']).count() != 5].index)

# race 上的不同
info.loc[complete, 'race'].value_counts()
info.loc[null_complete, 'race'].value_counts()   # non complete 白人多
# 画饼图
fig = plt.figure()
labels = ['White', 'Black', 'Asian', 'Pacific', 'Other']
X =  [44,14,5,1,3]
plt.pie(X,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("Completers race chart")
 
fig = plt.figure()
labels = ['White', 'Black', 'Asian', 'Pacific', 'Other']
X =  [47,12,6,1,1]
plt.pie(X,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("non-Completers race chart")
 

# sex 上的不同
info.loc[complete, 'sex'].value_counts()   # 1 是女生
info.loc[null_complete, 'sex'].value_counts()   # non complete 白人多
# 画饼图
fig = plt.figure()
labels = ['female', 'male', 'unknown']
X =  [50,16,1]
plt.pie(X,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("Completers sex chart")
 
fig = plt.figure()
labels = ['female', 'male', 'unknown']
X =  [54,13,0]
plt.pie(X,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("non-Completers sex chart")
 
# age 上的不同
info = data.groupby('patid').mean() 
info.loc[complete].mean()
info.loc[null_complete].mean()
plt.hist(info.loc[null_complete, 'age'], color = 'yellow', label = 'non complete', alpha =0.5)
plt.hist(info.loc[complete, 'age'],  color = 'r', label = 'complete', alpha =0.5)
plt.legend()
plt.title('age distribution')
plt.show()   

# blood pressure 上的不同
info = data.groupby('patid').mean() 
plt.hist(info.loc[null_complete, 'sbp'], color = 'yellow', label = 'non complete', alpha =0.5)
plt.hist(info.loc[complete, 'sbp'],  color = 'r', label = 'complete', alpha =0.5)
plt.legend()
plt.title('blood pressure distribution')
plt.show()   

# cpk 上的不同
info = data.groupby('patid').mean() 
plt.hist(info.loc[null_complete, 'cpk'], color = 'yellow', label = 'non complete', alpha =0.5)
plt.hist(info.loc[complete, 'cpk'],  color = 'r', label = 'complete', alpha =0.5)
plt.legend()
plt.title('cpk distribution')
plt.show()   

# haq 上的不同
info = data.groupby('patid').mean() 
plt.hist(info.loc[null_complete, 'haq'], color = 'yellow', label = 'non complete', alpha =0.5)
plt.hist(info.loc[complete, 'haq'],  color = 'r', label = 'complete', alpha =0.5)
plt.legend()
plt.title('haq distribution')
plt.show()   


# 或者可以直接用pandas画图
"""
ax = info.loc[null_complete, 'sbp'].plot(kind = 'hist', alpha = 0.3)  ax 参数可以让两个在一张图上
info.loc[complete, 'sbp'].plot(kind = 'hist', alpha = 0.3, ax =ax)
"""

# weight
plt.hist(info.loc[null_complete, 'weight'], color = 'yellow', label = 'non complete')
plt.hist(info.loc[complete, 'weight'],  color = 'purple', label = 'complete', alpha =0.3)
plt.legend()
plt.show()  


########## 观察大小计量 group
info = data.groupby('patid').mean() 
# blood pressure 上的不同
import seaborn as sns
g = sns.lmplot(x= 'times', y='sbp', hue = 'group', data = data)
# title
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)


# weight
import seaborn as sns
g = sns.lmplot(x= 'times', y='weight', hue = 'group', data = data) # most same
# title
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)

# cpk
g = sns.lmplot(x= 'times', y='cpk', hue = 'group', data = data) # most same
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)

# haq
g = sns.lmplot(x= 'times', y='haq', hue = 'group', data = data) # most same
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)


# nooffalls
g = sns.lmplot(x= 'times', y='nooffalls', hue = 'group', data = data) # most same
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)
    
# B 问
    # haq 有没有变化， 1） 用1 and 5times的人， 2） complter only， 3） 所有人（如果没有time = 5 ，）
    # 4）加减10天，没到就不算了（ 都用R）  5） 所有人 geeglm

# 用1 and 5times的人
import numpy as np
new1 = list(info[data.groupby('patid').times.max() == 5].index)
new1 = data[data.patid.isin(new1)]
new1 = new1.groupby(['patid', 'times']).mean()
new1 = new1.reset_index(level = ['patid', 'times']) # multi index转成col
# 用geeglm 做
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
fam = sm.families.Gaussian()
ex = sm.cov_struct.Exchangeable()
model1 = sm.GEE.from_formula("haq ~ group + times +group*times", groups="patid",
                      data=new1, family=fam, cov_struct=ex).fit()
print(model1.summary())

# haq 变化
g = sns.lmplot(x= 'times', y='haq', hue = 'group', data = new1) # most same
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)


# complete only
new2 = data[data.patid.isin(complete)]
new2 = new2.groupby(['patid', 'times']).mean()
new2 = new2.reset_index(level = ['patid', 'times']) # multi index转成col
# 用geeglm 做
fam = sm.families.Gaussian()
ex = sm.cov_struct.Exchangeable()
model2 = sm.GEE.from_formula("haq ~ group + times +group*times", groups="patid",
                      data=new2, family=fam, cov_struct=ex).fit()
print(model2.summary())

# haq 变化
g = sns.lmplot(x= 'times', y='haq', hue = 'group', data = new2) # most same
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)

# 最后一天改成times5 
def change(df):
    if df.iloc[len(df)-1, -1] != 5:
        df.iloc[len(df)-1, -1] =5
    return (df)

new5 = data.groupby('patid').apply(change) # apply 到一个新的df， 就没有level啦（因为已经是新的df了）
model5 = sm.GEE.from_formula("haq ~ group + times +group*times", groups="patid",
                      data=new5, family=fam, cov_struct=ex).fit()
print(model3.summary())
# haq 变化
g = sns.lmplot(x= 'times', y='haq', hue = 'group', data = new5) # most same
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)


# 对所有人
new3 = data.groupby(['patid', 'times']).mean()  #这种情况下就存在 level
new3 = new3.reset_index(level = ['patid', 'times']) # multi index转成col
model3 = sm.GEE.from_formula("haq ~ group + times +group*times", groups="patid",
                      data=new3, family=fam, cov_struct=ex).fit()
print(model3.summary())

# haq 变化
g = sns.lmplot(x= 'times', y='haq', hue = 'group', data = new3) # most same
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)

# within 10 days of the scheduled visit date 必须守时 (重新定义)
# 1) 先转成 距离天数
data_new  = origin.groupby(by = 'patid').apply(lambda x: x.sort_values('visitdat'))
data_new.reset_index(drop=True, inplace=True)
data_new = data_new.dropna(subset=['visitdat'])
data_new = data_new.groupby('patid').apply(minus)
# 转换为第几次
def check_new(time):
    if time == pd.Timedelta(0,'D'):
        times = 1
    elif  pd.Timedelta(172, 'D')< time < pd.Timedelta(192, 'D'):
        times = 2
    elif  pd.Timedelta(355, 'D')<time< pd.Timedelta(375, 'D'):
        times = 3
    elif pd.Timedelta(537, 'D') < time < pd.Timedelta(557, 'D'):
        times = 4
    elif pd.Timedelta(720, 'D') < time < pd.Timedelta(740, 'D'):
        times = 5
    else:
        times = 0
    return (times)
data_new['times'] = list(map(check_new, data_new['times'])) # map (对一系列数应用同一个function， return一个list)
new4 = data_new[data_new['times'] != 0]
patid_4 = list(info[data_new.groupby('patid').times.max() != 1].index)   # 符合条件的patid
new4 = new4[new4.patid.isin(patid_4)]
model4 = sm.GEE.from_formula("haq ~ group + times +group*times", groups="patid",
                      data=new4, family=fam, cov_struct=ex).fit()
print(model4.summary())

# haq 变化
g = sns.lmplot(x= 'times', y='haq', hue = 'group', data = new4) # most same
new_title = 'group'
g._legend.set_title(new_title)
# replace labels
new_labels = ['low dose', 'high dose']
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)

##################  到3题啦！
# 30% haq 算大， 其他算小， 用new1
q3_1 = new1[new1.times.isin([1,5])] # 仅留下拥有第一次和第五次记录的所有records
q3_1.reset_index(drop=True, inplace=True)
def check_2(df):
    if (df.iloc[1, 11] < 0.7*df.iloc[0, 11]):
        df['haq_level'] = 1
    else:
        df['haq_level'] = 0
    return (df)

q3_2 = q3_1.groupby('patid').apply(check_2)  # 二分类问题

# 做 卡方检验
q3_2.groupby(['group','haq_level']).count().patid   # 做个一个表
from scipy.stats import chi2_contingency
d = np.array([[28, 26], [22, 16]])
chi2_contingency(d)    # pvalue  0.7185446706976821
# 做LR model
import statsmodels.api as sm
q3_2_2 = q3_2.groupby('patid').mean()
log_2 = sm.Logit(np.array(q3_2_2['haq_level']).reshape(-1,1), sm.add_constant(np.array(q3_2_2['group']).reshape(-1,1))).fit()
log_2.summary()

import statsmodels.api as sm
q3_2_2 = q3_2.groupby('patid').mean()
log_2 = sm.Logit(np.array(q3_2_2['haq_level']).reshape(-1,1), sm.add_constant(np.array(q3_2_2[['group','cardrate','sbp','cpk']]))).fit()
log_2.summary()


# 三分类
def check_3(df):
    if (df.iloc[0, 11]-df.iloc[1, 11] > 1):
        df['haq_level'] = 2
    elif (df.iloc[0, 11]-df.iloc[1, 11] > 0.5):
        df['haq_level'] = 1
    else:
        df['haq_level'] = 0
    return (df)

q3_3 = q3_1.groupby('patid').apply(check_3)  # 三分类问题

# 做卡方检验
q3_3.groupby(['group','haq_level']).count().patid   # 做个一个表
from scipy.stats import chi2_contingency
d = np.array([[4, 36, 14], [0, 26,12]])
chi2_contingency(d)    # pvalue  0.21468001775625306

# 做三分类的LR ( 用NN)
import statsmodels.api as sm
q3_3_2 = q3_3.groupby('patid').mean()
log_3 = sm.MNLogit(endog = np.array(q3_3_2['haq_level']),exog = sm.add_constant(np.array(q3_3_2['group']))).fit()
log_3.summary()  # p-value


import statsmodels.api as sm
q3_3_2 = q3_3.groupby('patid').mean()
log_3 = sm.MNLogit(endog = np.array(q3_3_2['haq_level']),exog = sm.add_constant(np.array(q3_3_2[['group','cardrate','sbp','cpk']]))).fit()
log_3.summary()  # p-value


# 对于第二题，加上多的col后会有不一样吗！
# 先看二分类， 会有影响吗？
#再看三分类，会有影响吗？？
# 做LR model
import statsmodels.api as sm
q3_2_2 = q3_2.groupby('patid').mean()
log_2 = sm.Logit(np.array(q3_2_2['haq_level']).reshape(-1,1), sm.add_constant(np.array(q3_2_2['group']).reshape(-1,1))).fit()
log_2.summary()

# 做第四题啦
#预处理
data_4  = data.dropna()
data_4 = data_4[data_4.times == 1]
data_4 = data_4.reset_index()
data_4 = data_4.drop(['index','patid','visitdat','times'], axis = 1)  #扔掉cols
# 将race转换成onehot encoder
# 1. INSTANTIATE
# import preprocessing from sklearn
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
# 2. FIT
enc.fit(np.array(data_4.race).reshape(-1,1))
# 3. Transform
onehotlabels = enc.transform(np.array(data_4.race).reshape(-1,1)).toarray()
data_4['race'] = onehotlabels[:, 0]
data_4['race_1'] = onehotlabels[:, 1]
data_4['race_2'] = onehotlabels[:, 2]
X = data_4.drop(['nooffalls'], axis = 1)
y = data_4.nooffalls.values
# MLR model
import statsmodels.api as sm
from scipy import stats
dis = sm.add_constant(X)
y = y.reshape(len(dis),1)   #转换成dim相同的y
est = sm.OLS(y, np.array(dis))   # 先y，再X
est2 = est.fit()
est2.summary()

# LR model
# import statsmodels.api as sm
log_4 = sm.MNLogit(np.array(data_4.nooffalls.values),X).fit()
log_4.summary()  # p-value




