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
    mean_q = float(f.readline())
    mean_qq = float(f.readline())
    var_q = float(f.readline())
    var_qq = float(f.readline())

df_clt = pd.read_csv("exp_clt.csv")
sns.displot(df_clt,x="err",hue="T",col="var",kind="kde")
plt.show()

(df_clt[(df_clt["var"] == "q²") & (df_clt["T"] == 50)]["val"]).var()

df_q = df_clt[df_clt["var"] == "q"]
df_q["rescaled"] = np.sqrt(df_q["T"] * 1e-2 / var_q) * (df_q["val"] - mean_q)

df_qq = df_clt[df_clt["var"] == "q²"]
df_qq["rescaled"] = np.sqrt(df_qq["T"] * 1e-2 / var_qq) * (df_qq["val"] - mean_qq)

fig,axs = plt.subplots(1,2,figsize=(10,5),layout="tight")
sns.kdeplot(df_q,x="rescaled",hue="T",common_norm = False,ax=axs[0],palette=sns.color_palette())
axs[0].plot(np.linspace(-3,3),1/np.sqrt(2* np.pi) * np.exp(-np.linspace(-3,3)**2/2),"k:")
axs[0].set_ylabel("EPDF")
axs[0].set_xlabel("Residual Error")
axs[0].set_title("A")
sns.kdeplot(df_qq,x="rescaled",hue="T",common_norm = False,ax=axs[1],palette=sns.color_palette())
axs[1].set_xlabel("Residual Error")
axs[1].set_title("B")
axs[1].plot(np.linspace(-3,3),1/np.sqrt(2* np.pi) * np.exp(-np.linspace(-3,3)**2/2),"k:")
fig.savefig("fig/epdf.pdf")
plt.show()


