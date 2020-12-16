import pandas as pd

ppTD = pd.read_csv("./osim-rl/osim/data/stairs_down.csv", index_col=False)
workingTD = pd.read_csv("./osim-rl/osim/data/125-FIX.csv", index_col=False)

df = pd.DataFrame(None, columns=list(workingTD.columns))

for col in list(workingTD.columns):
    if col in list(ppTD.columns):
        df[col] = ppTD[col]
    else:
        df[col] = workingTD[col]

print(df.head())
df.to_csv("./stairs_down_FULL.csv", index=False)