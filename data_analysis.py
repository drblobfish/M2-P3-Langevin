import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig,axs = plt.subplots(1,2,figsize=(10,5))

df1 = pd.read_csv("exp1_epsilon.csv")
tidy1 = pd.melt(df1,id_vars="epsilon",value_name = "Asymptotic variance", var_name = "phi")
g = sns.lineplot(tidy1,x="epsilon",y="Asymptotic variance",hue = "phi",ax=axs[0])
g.loglog()

df2 = pd.read_csv("exp2_gamma.csv")
tidy2 = pd.melt(df2,id_vars="gamma",value_name = "Asymptotic variance", var_name = "phi")
g2 = sns.lineplot(tidy2,x="gamma",y="Asymptotic variance",hue = "phi", ax = axs[1])
g2.loglog()
fig.savefig("fig/fig_variance.pdf")
fig.savefig("fig/fig_variance.png")

with open("exp_clt_var.txt","r") as f:
    var_q = float(f.readline())
    var_qq = float(f.readline())

df_clt = pd.read_csv("exp_clt.csv")
sns.displot(df_clt,x="err",hue="T",col="var",kind="kde")
plt.show()
