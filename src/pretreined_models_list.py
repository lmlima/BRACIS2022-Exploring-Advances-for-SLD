import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/results/results-imagenet.csv")
df[['tipo', 'extra']] = df.model.str.replace("tf_", "").str.split('_', 1, expand=True)
selected_pretreined_model = df.query('param_count<50').sort_values('top1', ascending=False).drop_duplicates(subset=['tipo'])
selected_pretreined_model_str = selected_pretreined_model.model[:20].str.cat(sep=' ')
print(selected_pretreined_model_str)
