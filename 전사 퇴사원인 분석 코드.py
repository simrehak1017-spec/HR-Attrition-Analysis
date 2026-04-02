import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("HR-Employee-Attrition.csv")
df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})
df["OverTime"] = df["OverTime"].map({"Yes":1,"No":0})
df = df.drop(columns=["Over18","EmployeeNumber","EmployeeCount","StandardHours","YearsInCurrentRole","YearsWithCurrManager"])
df = pd.get_dummies(df, columns=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus"], drop_first=True, dtype=int)
print(df.corr(numeric_only=True)["Attrition"].sort_values(ascending=False))

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[["Age","TotalWorkingYears","MonthlyIncome","JobLevel"]].astype(float)

vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

# ---------------- 로지스틱 회귀분석 ----------------

import numpy as np
import statsmodels.api as sm

X_logit = df[["OverTime","MaritalStatus_Single","JobRole_Sales Representative","TotalWorkingYears","MonthlyIncome"]].astype(float)
y = df["Attrition"].astype(int)

X_logit = sm.add_constant(X_logit)

model = sm.Logit(y, X_logit)
result = model.fit()
print(result.summary())

odds_ratio = pd.DataFrame({"Variable": result.params.index,"Coefficient": result.params.values,"Odds_Ratio": np.exp(result.params.values),"P_value": result.pvalues.values})

print("\nOdds Ratio Table")
print(odds_ratio)

significant = odds_ratio[odds_ratio["P_value"] < 0.05]
print("\nSignificant Variables (p < 0.05)")
print(significant)
