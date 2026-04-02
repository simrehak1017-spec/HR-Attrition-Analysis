import pandas as pd

df = pd.read_csv("HR-Employee-Attrition.csv")

df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})
df["OverTime"] = df["OverTime"].map({"Yes":1, "No":0})

df = df.drop(columns=["Over18","EmployeeCount","StandardHours","EmployeeNumber"])
df = pd.get_dummies(df,columns=["BusinessTravel","Department","EducationField","Gender","MaritalStatus"],drop_first=True,dtype=int)

df_salesrep = df[df["JobRole"] == "Sales Representative"].copy()

corr = df_salesrep.corr(numeric_only=True)["Attrition"].sort_values(ascending=False)
print(corr)

import statsmodels.api as sm
X = df_salesrep[["OverTime","BusinessTravel_Travel_Frequently","WorkLifeBalance","JobInvolvement","YearsInCurrentRole","JobSatisfaction"]].astype(float)
y = df_salesrep["Attrition"]
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(model.summary())

import numpy as np
import pandas as pd

# 1. 로지스틱 회귀 결과표를 OR 기준으로 변환
params = model.params
conf = model.conf_int()
pvals = model.pvalues

or_table = pd.DataFrame({"coef": params,"odds_ratio": np.exp(params),"p_value": pvals,"CI_lower": np.exp(conf[0]),"CI_upper": np.exp(conf[1])})

print("\n=== Odds Ratio Table ===")
print(or_table.round(3))


# 2. 초과근무 여부에 따른 주요 변수 평균 비교 
import matplotlib.pyplot as plt

group_analysis = df_salesrep.groupby("BusinessTravel_Travel_Frequently")[["JobInvolvement","JobSatisfaction","YearsInCurrentRole"]].mean()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

metrics = ["JobInvolvement", "JobSatisfaction", "YearsInCurrentRole"]
titles = ["Average Job Involvement","Average Job Satisfaction","Average Years in Current Role"]
for i, metric in enumerate(metrics):
    values = group_analysis[metric]
    bars = axes[i].bar(["Travel No", "Travel Yes"], values)
    
    axes[i].set_title(titles[i])
    axes[i].set_ylabel("Average Value")
for bar in bars: 
    y = bar.get_height()
    axes[i].text(bar.get_x() + bar.get_width()/2, y + 0.03, f"{y:.2f}",
                 ha="center", va="bottom")

plt.tight_layout()
plt.show()