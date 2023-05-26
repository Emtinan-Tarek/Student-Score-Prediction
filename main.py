import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel(r'C:\SA Proj\SA DataSet\Admission_Predict.xlsx')
print(df['Graduate Record Examination Score (GRE) Score'].describe())
print(df['Test of English as a Foreign Language (TOEFL) Score'].describe())
print(df['Statement of Purpose Strength'].describe())
print(df['Letter of Recommendation Strength'].describe())
print(df['Cummulative GPA'].describe())
print(df['Research'].describe())
print(df['Chance of Admit'].describe())
#Detecting outliers
def detect_outlier(data_1):
    outliers = []
    q1, q3 = np.percentile(data_1, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for y in data_1:
        if y < lower_bound or y > upper_bound:
            outliers.append(y)

    if not outliers:
        return "No Outliers"
    else:
        return outliers
print('GRE Score Outliers:')
print(detect_outlier(df['Graduate Record Examination Score (GRE) Score']))
print('TOEFL Score Outliers:')
print(detect_outlier(df['Test of English as a Foreign Language (TOEFL) Score']))
print('University Rating Outliers:')
print(detect_outlier(df['University Rating']))
print('SOP Outliers:')
print(detect_outlier(df['Statement of Purpose Strength']))
print('CGPA Outliers:')
print(detect_outlier(df['Cummulative GPA']))
print('LOR Outliers:')
print(detect_outlier(df['Letter of Recommendation Strength']))
print('Chance Of Admission Outliers:')
print(detect_outlier(df['Chance of Admit']))

#Our Dataset's Boxplots
plt.title("TOEFL Score ")
df['Test of English as a Foreign Language (TOEFL) Score'].plot(kind='box', patch_artist=True, color=dict(boxes='pink', whiskers='black', medians='blue', caps='yellow'))
plt.show()

plt.title("GRE Score")
df['Graduate Record Examination Score (GRE) Score'].plot(kind='box', patch_artist=True, color=dict(boxes='green', whiskers='black', medians='yellow', caps='black'))
plt.show()

plt.title("University Ratings of Students")
df['University Rating'].plot(kind='box', patch_artist=True, color=dict(boxes='red', whiskers='orange', medians='black', caps='pink'))
plt.show()

plt.title("Statement of Purpose Strength")
df['Statement of Purpose Strength'].plot(kind='box', patch_artist=True, color=dict(boxes='blue', whiskers='brown', medians='white', caps='pink'))
plt.show()

plt.title("Letter of Recommendation Strength")
df['Letter of Recommendation Strength'].plot(kind='box', patch_artist=True, color=dict(boxes='yellow', whiskers='brown', medians='blue', caps='pink'))
plt.show()

plt.title("Chance of Admit ")
df['Chance of Admit'].plot(kind='box', patch_artist=True, color=dict(boxes='purple', whiskers='yellow', medians='blue', caps='pink'))
plt.show()

# Frequency Tables
print('GRE Frequency Table:')
grouped_freq_GRE = df['Graduate Record Examination Score (GRE) Score'].value_counts(bins=10).sort_index()
print(grouped_freq_GRE)
print()

print('TOEFL Score Frequency Table:')
grouped_freq_toefl = df['Test of English as a Foreign Language (TOEFL) Score'].value_counts(bins=10).sort_index()
print(grouped_freq_toefl)
print()

print('CGPA Frequency Table:')
grouped_freq_cgpa = df['Cummulative GPA'].value_counts(bins=10).sort_index()
print(grouped_freq_cgpa)
print()

print('Chance of Admission Frequency Table:')
grouped_freq_admit = df['Chance of Admit'].value_counts(bins=10).sort_index()
print(grouped_freq_admit)
print()

print('University Rating Frequency Table:')
print(pd.crosstab(index=df['University Rating'],columns='Frequency',))
print()

print('SOP Frequency Table:')
print(pd.crosstab(index=df['Statement of Purpose Strength'], columns='Frequency',))
print()

print('LOR Frequency Table:')
print(pd.crosstab(index=df['Letter of Recommendation Strength'], columns='Frequency'))
print()

print('Research Frequency Table:')
print(pd.crosstab(index=df['Research'], columns='Frequency'))
print()

plt.figure(figsize=(12, 7))

# GRE Score Histogram
plt.subplot(3, 3, 1)
plt.hist(df['Graduate Record Examination Score (GRE) Score'], edgecolor='black')
plt.title('GRE Score of Students')
plt.xlabel('GRE Score')
plt.ylabel('Number of Students')

# TOEFL Score Histogram
plt.subplot(3, 3, 2)
plt.hist(df['Test of English as a Foreign Language (TOEFL) Score'], edgecolor='black')
plt.title('TOEFL Score of Students')
plt.xlabel('TOEFL Score')
plt.ylabel('Number of Students')

# University Rating Bar Chart
plt.subplot(3, 3, 3)
s = df['University Rating'].value_counts().head(5).sort_index()
plt.title("University Ratings of Students")
s.plot(kind='bar', linestyle='dashed', linewidth=5)

plt.xlabel("University Rating")
plt.ylabel("Number of Students")

# SOP Strength Bar Chart
plt.subplot(3, 3, 4)
s = df['Statement of Purpose Strength'].value_counts().head(9).sort_index()
plt.title("Statement Of Purpose Strength")
s.plot(kind='bar', linestyle='dashed', linewidth=5)
plt.xlabel("SOP Strength")
plt.ylabel("Number of Students")

# LOR Strength Bar Chart
plt.subplot(3, 3, 5)
s = df['Letter of Recommendation Strength'].value_counts().head(9).sort_index()
plt.title("Letter of Recommendation Strength")
s.plot(kind='bar', linestyle='dashed', linewidth=5)
plt.xlabel("LOR Strength")
plt.ylabel("Number Of Students")

# CGPA Score Histogram
plt.subplot(3, 3, 6)
plt.hist(df['Cummulative GPA'], edgecolor='black')
plt.title('CGPA of Students')
plt.xlabel('CGPA')
plt.ylabel('Number of Students')

# Chance of Admission Histogram
plt.subplot(3, 3, 7)
plt.hist(df['Chance of Admit'], edgecolor='black')
plt.title('Chance Of Admission of Students')
plt.xlabel('Chance of Admission')
plt.ylabel('Number of Students')
plt.tight_layout(pad=3.0, w_pad=5.0, h_pad=2.0)

# Research Experience Bar Chart
plt.subplot(3, 3, 8)
s = df['Research'].value_counts().head(2).sort_index()
plt.title("Research Experience of Students")
s.plot(kind='bar', linestyle='dashed', linewidth=5)

plt.xlabel("Research Experience")
plt.ylabel("Number of Students")

plt.suptitle('Histograms and Bar Charts')
plt.show()


plt.hist(df['Graduate Record Examination Score (GRE) Score'], edgecolor='black')

median_GRE = 317
plt.axvline(median_GRE, color='red', label='GRE Median')
mean_GRE = 316
plt.axvline(mean_GRE, color='orange', label='GRE Mean')
mode_GRE = mode(df['Graduate Record Examination Score (GRE) Score'])
plt.axvline(mode_GRE, color='yellow', label='GRE Mode')
plt.legend()
plt.title('GRE Score of Students')
plt.xlabel('GRE Score')
plt.ylabel('Number of Students')
plt.show()


plt.hist(df['Test of English as a Foreign Language (TOEFL) Score'], edgecolor='black')

median_TOEFL = 107
plt.axvline(median_TOEFL, color='red', label='TOEFL Median')

mean_TOEFL = 107.41
plt.axvline(mean_TOEFL, color='orange', label='TOEFL Mean')

mode_TOEFL = mode(df['Test of English as a Foreign Language (TOEFL) Score'])
plt.axvline(mode_TOEFL, color='yellow', label='TOEFL Mode')
plt.legend()
plt.title('TOEFL Score of Students')
plt.xlabel('TOEFL Score')
plt.ylabel('Number of Students')
plt.show()

plt.hist(df['Cummulative GPA'], edgecolor='black')

median_cgpa = 8.6
plt.axvline(median_cgpa, color='red', label='CGPA Median')

mean_cgpa = 8.59
plt.axvline(mean_cgpa, color='orange', label='CGPA Mean')

mode_cgpa = mode(df['Cummulative GPA'])
plt.axvline(mode_cgpa, color='yellow', label='CGPA Mode')
plt.legend()
plt.title('CGPA of Students')
plt.xlabel('CGPA')
plt.ylabel('Number of Students')
plt.show()


plt.hist(df['Chance of Admit'], edgecolor='black')

median_chanceOfAdmit = df['Chance of Admit'].median()
plt.axvline(median_chanceOfAdmit, color='red', label='Chance Of Admit Median')

mean_chanceOfAdmit = df['Chance of Admit'].mean()
plt.axvline(mean_chanceOfAdmit, color='orange', label='Chance Of Admit Mean')

mode_chanceOfAdmit = mode(df['Chance of Admit'])
plt.axvline(mode_chanceOfAdmit, color='yellow', label='Chance of Admit Mode')
plt.legend()
plt.title('Chance Of Admission of Students')
plt.xlabel('Chance of Admission')
plt.ylabel('Number of Students')
plt.show()

#Scatterplot and Linear Regression for Effect of GRE on Chance of Admission
x = np.array(df['Graduate Record Examination Score (GRE) Score'])
y=np.array(df['Chance of Admit'])
plt.title("Effect of GRE Score on Students' Chance of Admission",weight="bold",size=8)
plt.xlabel("GRE Score (out of 340)", weight="bold",color='darkred',size=7)
plt.ylabel("Chance of Admission",weight="bold",color='darkred',size=7)
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(round(r,2))
x_mean = np.mean(x)
y_mean = np.mean(y)
Sx = np.std(x)
Sy = np.std(y)
b1 = r*(Sy/Sx)
b0 = y_mean-(b1*x_mean)
print("y^="+str(round(b0, 2))+"+"+str(b1)+"x")
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))

plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()

#Scatterplot and Linear Regression for Effect of TOEFL on Chance of Admission
z = np.array(df['Test of English as a Foreign Language (TOEFL) Score'])
q = np.array(df['Chance of Admit'])
plt.title("Effect of TOEFL Score on Student's Chance of Admission",weight="bold",size=8)
plt.xlabel("TOEFL Score (out of 120)", weight="bold",color='darkcyan',size=7)
plt.ylabel("Chance of Admission", weight="bold",color='darkcyan',size=7)
slope, intercept, r, p, std_err = stats.linregress(z, q)
print(round(r,2))
z_mean = np.mean(z)
q_mean = np.mean(q)
Sz = np.std(z)
Sq = np.std(q)
b1 = r*(Sq/Sz)
b0 = y_mean-(b1*z_mean)
print("y^="+str(round(b0, 2))+"+"+str(b1)+"x")
def myfunc(z):
  return slope * z + intercept
mymodel = list(map(myfunc, z))
plt.scatter(z,q,color='#800080')
plt.plot(z,mymodel,color='#800080')
plt.show()

#Scatterplot and Linear Regression for Effect of University Rating on Chance of Admission
x=np.array(df['University Rating'])
y=np.array(df['Chance of Admit'])
plt.title("Effect of University Rating on Student's Chance of Admission",weight="bold",size=8)
plt.xlabel("University Rating (from 5)",weight="bold",color='black',size=7)
plt.ylabel("Chance of Admission",weight="bold",color='black',size=7)
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(round(r,2))
x_mean = np.mean(x)
y_mean = np.mean(y)
Sx = np.std(x)
Sy = np.std(y)
b1 = r*(Sy/Sx)
b0 = y_mean-(b1*x_mean)
print("y^="+str(round(b0, 2))+"+"+str(b1)+"x")
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x,y,color='#4B0082')
plt.plot(x,mymodel,color='#4B0082')
plt.show()

#Scatterplot and Linear Regression for Effect of Statement of Purpose Strength on Chance of Admission
x=np.array(df['Statement of Purpose Strength'])
y=np.array(df['Chance of Admit'])
plt.title("Effect of Statement of Purpose on Student's Chance of Admission",weight="bold",size=8)
plt.xlabel("Statement of Purpose Strength (SOP) ",weight="bold",color='cadetblue',size=7)
plt.ylabel("Chance of Admission",weight="bold",color='cadetblue',size=7)
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(round(r,2))
x_mean = np.mean(x)
y_mean = np.mean(y)
Sx = np.std(x)
Sy = np.std(y)
b1 = r*(Sy/Sx)
b0 = y_mean-(b1*x_mean)
print("y^="+str(round(b0, 2))+"+"+str(b1)+"x")
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y, color='#B22222')
plt.plot(x, mymodel, color='#B22222')
plt.show()

#Scatterplot and Linear Regression for Effect of Letter of Recommendation Strength on Chance of Admission
x = np.array(df['Letter of Recommendation Strength'])
y = np.array(df['Chance of Admit'])
plt.title("Effect of Letter of Recommendation Strength on Chance of Admission",weight="bold",size=8)
plt.xlabel("Letter of Recommendation (LOR) ",weight="bold",color='#000080',size=7)
plt.ylabel("Chance of Admission",weight="bold",color='#000080',size=7)
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(round(r,2))
x_mean = np.mean(x)
y_mean = np.mean(y)
Sx = np.std(x)
Sy = np.std(y)
b1 = r*(Sy/Sx)
b0 = y_mean-(b1*x_mean)
print("y^="+str(round(b0, 2))+"+"+str(b1)+"x")
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x,y,color='#66CDAA')
plt.plot(x,mymodel,color='#66CDAA')
plt.show()

#Scatterplot and Linear Regression for Effect of CGPA on Chance of Admission
x=np.array(df['Cummulative GPA'])
y=np.array(df['Chance of Admit'])
plt.title("Effect of Cumulative GPA (CGPA) on Chance of Admission",weight="bold",size=8)
plt.xlabel("Cumulative GPA (CGPA) ",weight="bold",color='black',size=7)
plt.ylabel("Chance of Admission",weight="bold",color='black',size=7)
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(round(r,2))
x_mean = np.mean(x)
y_mean = np.mean(y)
Sx = np.std(x)
Sy = np.std(y)
b1 = r*(Sy/Sx)
b0 = y_mean-(b1*x_mean)
print("y^="+str(round(b0, 2))+"+"+str(b1)+"x")
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x,y,color='darkorange')
plt.plot(x,mymodel,color='darkorange')
plt.show()

#User-Interaction to Predict Values Using Regression for GRE Score
print("Predicting chance of admit of a student and deducing the error in prediction:")
ActualGRE = int(input(print("Please enter your actual GRE score: ")))
ActualChanceAdmit = float(input(print("Please enter your actual chance of admission: ")))
EstimatedChance = -2.44+(0.01*ActualGRE)
print("Your estimated chance of admit is " + str(round(EstimatedChance, 2)))
Error = abs(ActualChanceAdmit - EstimatedChance)
print("The error in prediction is " + str(round(Error, 2)))

#Correlation Matrix-Heat Map
df.columns = ['Student No', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP Strength', 'LOR Strength', 'CGPA', 'Research', 'Chance of Admit']
correlation_heat_map = df.corr()
axis_corr = sns.heatmap(correlation_heat_map,vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(50, 500, n=500),square=True)
plt.show()

#Clustering
plt.scatter(df['TOEFL Score'], df['Chance of Admit'])
plt.xlabel('TOEFL Score')
plt.ylabel('Chance of Admit')
plt.show()
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['TOEFL Score', 'Chance of Admit']])
print(y_predicted)
df['cluster'] = y_predicted

print(km.cluster_centers_)
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
plt.scatter(df1['TOEFL Score'], df1['Chance of Admit'], color='green')
plt.scatter(df2['TOEFL Score'], df2['Chance of Admit'], color='red')
plt.scatter(df3['TOEFL Score'], df3['Chance of Admit'], color='black')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.xlabel('TOEFL Score')
plt.ylabel('Chance of Admit')
print(plt.legend())
plt.show()

# Processing using MinMaxscaler
scaler = MinMaxScaler()
scaler.fit(df[['Chance of Admit']])
df['Chance of Admit'] = scaler.transform(df[['Chance of Admit']])
scaler.fit(df[['TOEFL Score']])
df['TOEFL Score'] = scaler.transform(df[['TOEFL Score']])
# df.head()
plt.scatter(df['TOEFL Score'], df['Chance of Admit'])
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['TOEFL Score', 'Chance of Admit']])
print(y_predicted)
df['cluster'] = y_predicted

print(km.cluster_centers_)
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
plt.scatter(df1['TOEFL Score'], df1['Chance of Admit'], color='green')
plt.scatter(df2['TOEFL Score'], df2['Chance of Admit'], color='red')
plt.scatter(df3['TOEFL Score'], df3['Chance of Admit'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.legend()
plt.show()

# Elbow plot
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['TOEFL Score', 'Chance of Admit']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()

print("Conclusion of the chance of admit depending on TOEFL Test Scores")
print()
print(str((df1['Student No'].count()/400)*100) + " percent of TOEFL students have a 30 to 80 percent chance of admit")

print(str((df2['Student No'].count()/400)*100)+" percent of TOEFL students have a 59 to 100 percent chance of admit")

print(str((df3['Student No'].count()/400)*100)+" percent of TOEFL students have no more than 55 percent chance of admit")

#logistic Regression


print('Logistic Regression')
print('Predicting Research from Chance of Admit')
X_train, X_test, y_train, y_test = train_test_split(df[['Chance of Admit']], df['Research'], test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
print('Test Cases : ')
print(X_test)
y_predicted = model.predict(X_test)
print('Percentage of a possible [ 1 ] of each testcase : ')
print(model.predict_proba(X_test)[:, 1])
print('Predicted values : ', y_predicted)
print('( 1 ) representing the student "did" a Research '
      '& ( 0 ) representing the student "did not" do a Research')
print('Test Accuracy : ', model.score(X_test, y_test))

sns.regplot(x='Chance of Admit', y='Research', data=df, logistic=True)
plt.title('Predicting Research from Chance of Admit using Logistic Regression')
plt.show()

cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.show()

print()

print('Predicting Chance of Admit from CGPA')
df.loc[df['Chance of Admit'] >= 0.5, 'Chance of Admit'] = 1
df.loc[df['Chance of Admit'] < 0.5, 'Chance of Admit'] = 0

X_train, X_test, y_train, y_test = train_test_split(df[['CGPA']], df['Chance of Admit'], test_size=0.1)

model = LogisticRegression()
model.fit(X_train, y_train)

print('Test Cases : ')
print(X_test)
y_predicted = model.predict(X_test)
print('Percentage of a possible [ 1 ] of each testcase : ')
print(model.predict_proba(X_test)[:, 1])
print('Predicted values : ', y_predicted)
print('( 1 ) representing Chance of Admit equal to or higher than 0.5% '
      '& ( 0 ) representing Chance of Admit lower than 0.5%')
print('Test Accuracy : ', model.score(X_test, y_test))

sns.regplot(x='CGPA', y='Chance of Admit', data=df, logistic=True)
plt.title('Predicting Chance of Admit from CGPA using Logistic Regression')
plt.show()

cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()