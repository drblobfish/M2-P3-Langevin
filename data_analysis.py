import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig,axs = plt.subplots(1,2)

df1 = pd.read_csv("exp1_epsilon.csv")
tidy1 = pd.melt(df1,id_vars="epsilon",value_name = "Asymptotic variance", var_name = "phi")
g = sns.lineplot(tidy1,x="epsilon",y="Asymptotic variance",hue = "phi",ax=axs[0])
g.loglog()

df2 = pd.read_csv("exp2_gamma.csv")
tidy2 = pd.melt(df2,id_vars="gamma",value_name = "Asymptotic variance", var_name = "phi")
g = sns.lineplot(tidy2,x="gamma",y="Asymptotic variance",hue = "phi", ax = axs[1])
g.loglog()
plt.show()
