import numpy as np
from numpy.random import randn
import pandas as pd
from pandas import Series, DataFrame as df
import scipy
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

CompX = pd.ExcelFile('CompanyX.xlsx')
existing = df(CompX.parse('Existing employees'))
left = df(CompX.parse('Employees who have left'))
left['status'] = 'left'
existing['status'] = 'existing'
# print(existing,left)

full_data = pd.concat([left, existing], axis=0)
print(df.nunique(full_data))
# print(full_data)
# print(full_data.describe())

# dept_data=df.pivot(full_data,'dept','status')
# Series.plot(existing['dept'],'pie','%1.1f%')
# plt.show()


full_data['satisfaction_level']=pd.cut(full_data['satisfaction_level'],10)
table = pd.crosstab(full_data['satisfaction_level'], full_data['status'])
print(table)
table_fig = table.plot(kind='barh', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Satisfaction Level with Status Differentiation',fontsize=20)
plt.ylabel('Satisfaction Level', fontsize=15)
table_fig.savefig('Satlevel_status.png')

full_data['last_evaluation']=pd.cut(full_data['last_evaluation'],10)
table = pd.crosstab(full_data['last_evaluation'], full_data['status'])
print(table)
table_fig = table.plot(kind='barh', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Last Evaluation with Status Differentiation',fontsize=20)
plt.ylabel('Last Evaluation',fontsize=15)
table_fig.savefig('lastEval_status.png')

full_data['average_montly_hours']=pd.cut(full_data['average_montly_hours'],10)
table = pd.crosstab(full_data['average_montly_hours'], full_data['status'])
print(table)
table_fig = table.plot(kind='barh', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Average Monthly Hours with Status Differentiation',fontsize=20)
plt.ylabel('Average Monthly Hours',fontsize=15)
table_fig.savefig('AvgHrs_status.png')

# full_data['last_evaluation']=pd.cut(full_data['last_evaluation'],10)
table = pd.crosstab(full_data['number_project'], full_data['status'])
print(table)
table_fig = table.plot(kind='bar', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Number of Projects with Status Differentiation',fontsize=20)
plt.xlabel('Number of Projects',fontsize=15)
table_fig.savefig('project_status.png')

table = pd.crosstab(full_data['time_spend_company'], full_data['status'])
print(table)
table_fig = table.plot(kind='bar', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Time Spent in the Company with Status Differentiation',fontsize=20)
plt.xlabel('Time Spent in the Company',fontsize=15)
table_fig.savefig('time_spent_status.png')

table = pd.crosstab(full_data['dept'], full_data['status'])
print(table)
table_fig = table.plot(kind='bar', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Departments with Status Differentiation',fontsize=20)
plt.xlabel('Departments',fontsize=15)
table_fig.savefig('dept_status.png')

table = pd.crosstab(full_data['salary'], full_data['status'])
print(table)
table_fig = table.plot(kind='bar', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Salary with Status Differentiation',fontsize=20)
plt.xlabel('Salary',fontsize=15)
table_fig.savefig('salary_status.png')

table = pd.crosstab(full_data['Work_accident'], full_data['status'])
print(table)
table_fig = table.plot(kind='bar', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Number of Work Accidents with Status Differentiation',fontsize=20)
plt.xlabel('Number of Work Accidents',fontsize=15)
table_fig.savefig('Accid_status.png')

table = pd.crosstab(full_data['promotion_last_5years'], full_data['status'])
print(table)
table_fig = table.plot(kind='bar', stacked=True, figsize=(10,10),fontsize=12).get_figure()
plt.title('Plot of Promotion in the last 5 years with Status Differentiation',fontsize=20)
plt.xlabel('Promotion in the last 5 years',fontsize=15)
table_fig.savefig('Lst5yrsPromo_status.png')


'''
Observations from Graphs
The Higher the Satisfaction, the more existing
The middle guys left the most - Time Spent
High Salary guys stayed
3-5 project stayed, highest number of project leaves
Last Eval, poorest stayed, mid stayed too
Sales, Support and Technical guys left the most.
'''

existing = df(CompX.parse('Existing employees'))
left = df(CompX.parse('Employees who have left'))
left['status'] = 1
existing['status'] = 0
existing1,existing2=np.split(existing,2,axis=0)
print(existing1.describe(),existing2.describe())
# print(existing,left)

full_data_predict = pd.concat([left, existing1], axis=0)
print(df.nunique(full_data_predict))

X_train, X_test, y_train, y_test = train_test_split(full_data_predict.iloc[:, 1:7],
                                                    full_data_predict['status'],test_size=0.2)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
# existing_employees_test=existing
print(knn.score(X_test,y_test))
prediction = knn.predict(existing.iloc[:, 1:7])
print(scipy.stats.describe(prediction))
print(np.unique(prediction,return_counts=True))
print(np.asarray(prediction))
# np.savetxt('txtnp.txt', prediction.describe(), delimiter=',')

'''
full_corr=full_data.corr()
full_corr.to_csv('corrFull.csv')
full_corr_heatmap=sns.heatmap(full_corr,xticklabels=full_corr.columns, yticklabels=full_corr.columns,
            annot=True,cmap=sns.diverging_palette(220,20,as_cmap=True)).get_figure()
full_corr_heatmap.savefig('full_corr.png')


corr_existing=df.corr(existing)
corr_existing.to_csv('corr_existing.csv')

print(left.describe())
print(existing.describe())
print(pd.isnull(left))
print(pd.isnull(existing))

print(df.nunique(existing))
print(df.nunique(left))

sns.pairplot(existing).savefig('existing1.png')
sns.pairplot(left).savefig('left.png')

sns.heatmap(existing, annot=True).get_figure().savefig('heatmap1.png')
sns.heatmap(left, annot=True).get_figure().savefig('heatmap2.png')
'''