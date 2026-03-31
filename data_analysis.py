import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("exp1_epsilon.csv")
tidy1 = pd.melt(df1,id_vars="epsilon",value_name = "Asymptotic variance", var_name = "phi")
sns.lineplot(tidy1,x="epsilon",y="Asymptotic variance",hue = "phi")
plt.show()
