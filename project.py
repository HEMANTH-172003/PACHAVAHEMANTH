import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("Data Science Jobs Salaries.csv")
df.info()
df.isnull().sum()
df.nunique()
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='experience_level')
plt.xlabel('Experience Level')
plt.ylabel('Count')
plt.title('Distribution of Work Experience Levels')
plt.show()
# Group the data by work year
df1 = df.groupby("work_year")["salary"].mean()

# Plot the average salary by work year
df1.plot(kind="bar")

# Add a title and labels
plt.title("Average salary by work year")
plt.xlabel("Work year")
plt.ylabel("Salary")

# Show the plot
plt.show()
df.hist()
plt.show()
print("Total value counts of the roles:-\n ",df["experience_level"].value_counts())
roles = ["SE", "MI", "EN", "EX"]
people = [2516, 805, 320, 114]
print(df["employment_type"].value_counts())
types=["FT","PT","CT","FL"]
no_people=[3718,17,10,10]
plt.pie(no_people, labels=types, autopct='%1.1f%%',explode = [0.3, 0, 0,0])
plt.title('Number of people by emp_type')
plt.axis('equal')
plt.show()
df["company_size"].value_counts()
company_numbers = [3153, 454, 148]
company_size = ["M", "L", "S"]
explode = [0.1, 0, 0]  
plt.pie(company_numbers, labels=company_size, explode=explode, autopct='%1.1f%%')
plt.axis('equal')
plt.title("Company size")
plt.show()
df["job_title"].value_counts()
job_title_salaries = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 20))
ax.barh(job_title_salaries.index, job_title_salaries.values)
ax.set_title('Average Salaries by Job Title')
ax.set_xlabel('Salary in USD')
ax.set_ylabel('Job Title')
plt.show()
sns.catplot(x="experience_level",y="salary_in_usd" ,kind="bar",data=df)   
plt.show()

sns.catplot(x="employment_type",y="salary_in_usd" ,kind="bar",data=df)   
plt.show()
sns.catplot(x="work_year",y="salary_in_usd" ,kind="bar",data=df)   
plt.show()
sns.catplot(x="company_size",y="salary_in_usd",kind="box",data=df)   
plt.title("Box plot grouped by company size")
plt.show()
sns.catplot(x="experience_level",y="salary_in_usd",hue="company_size" ,kind="box",data=df)   
plt.title("Box plot grouped by company size")
plt.show()
cat_list=[i for i in df.select_dtypes("object")]
cat_list
for i in cat_list:
    df[i] = df[i].factorize()[0]
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True,linewidths=0.7,cmap="viridis",fmt=".2f")
plt.show()
X=df.drop(["salary_in_usd"], axis = 1)
Y=df["salary_in_usd"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import neighbors
from sklearn.svm import SVR
dt=DecisionTreeRegressor()
dt.fit(X_train,Y_train)
y_predict = dt.predict(X_test)
dt.score(X_train,Y_train)
dt.score(X_test,Y_test)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(r2_score(Y_test, y_predict)*100)
print(mean_squared_error(Y_test, y_predict))
print(mean_absolute_error(Y_test, y_predict))
knn=KNeighborsRegressor().fit(X_train,Y_train)
ada=AdaBoostRegressor().fit(X_train,Y_train)
svm=SVR().fit(X_train,Y_train)
ridge=Ridge().fit(X_train,Y_train)
lasso=Lasso().fit(X_train,Y_train)
rf=RandomForestRegressor().fit(X_train,Y_train)
gbm=GradientBoostingRegressor().fit(X_train,Y_train)
models=[ridge,lasso,knn,ada,svm,rf,gbm]
def ML(Y,models):
    y_pred=models.predict(X_test)
    mse=mean_squared_error(Y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
    r2=r2_score(Y_test,y_pred)*100
    
    return mse,rmse,r2
for i in models:
    print("\n",i,"\n\nDifferent models success rate :",ML("salary_in_usd",i))
