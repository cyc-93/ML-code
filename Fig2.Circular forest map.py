import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

df = pd.read_csv("Fig2a-c-final.csv", dtype=str)
rename = {'Stress factor': 'Stress.factor', 'ln Resp.R': 'ln.Resp.R',
          'Var(ln Resp.R)': 'Var.ln.Resp.R'}
df.rename(columns=rename, inplace=True)
df['Stress.factor'] = df['Stress.factor'].str.strip()

for c in ['ln.Resp.R', 'Var.ln.Resp.R']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df.dropna(subset=['ln.Resp.R', 'Var.ln.Resp.R'], inplace=True)

df['se'] = np.sqrt(df['Var.ln.Resp.R'])
df['lower'] = df['ln.Resp.R'] - 1.96 * df['se']
df['upper'] = df['ln.Resp.R'] + 1.96 * df['se']

groups = ['Drought', 'Salinity', 'Plant disease']
color_map = {
    'Drought': '#de978e',
    'Salinity': '#90c286',
    'Plant disease': '#8bacc4'
}

df_sorted = df.sort_values(['Stress.factor', 'ln.Resp.R']).reset_index(drop=True)
blocks = []
for g in groups:
    blocks.append(df_sorted[df_sorted['Stress.factor'] == g])
    blocks.append(pd.DataFrame({col: [np.nan] for col in df_sorted.columns}))
df_final = pd.concat(blocks, ignore_index=True)
df_final['idx'] = np.arange(len(df_final))
df_final['angle'] = df_final['idx'] / len(df_final) * 2 * np.pi

lo, hi = -5, 5
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

for g in groups:
    seg = df_final[df_final['Stress.factor'] == g].dropna(subset=['ln.Resp.R'])
    if seg.empty: continue
    col = color_map[g]
    ax.plot(seg['angle'], np.clip(seg['ln.Resp.R'], lo, hi), color=col, lw=2)
    for _, r in seg.iterrows():
        low, up = max(r['lower'], lo), min(r['upper'], hi)
        ax.vlines(r['angle'], low, up, color=col, lw=0.6)
        if r['upper'] > hi:   ax.plot(r['angle'], hi, marker=r'$\uparrow$', color=col, ms=6, mew=0)
        if r['lower'] < lo:   ax.plot(r['angle'], lo, marker=r'$\downarrow$', color=col, ms=6, mew=0)

for r, s in [(-5, '-'), (-1, '-'), (0, '--'), (1, '-'), (5, '-')]:
    ax.plot(np.linspace(0, 2 * np.pi, 400), np.full(400, r), ls=s, lw=0.7, c='grey')

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticklabels([]);
ax.set_yticklabels([]);
ax.set_ylim(lo - 0.2, hi + 0.2);
ax.grid(False)

counts = df['Stress.factor'].value_counts().reindex(groups, fill_value=0)
pie_ax = inset_axes(ax, width='34%', height='34%', loc='center')
pie_ax.pie(counts, colors=[color_map.get(g, '#cccccc') for g in groups], startangle=90)
pie_ax.set_aspect('equal')

ax.set_title('Circular Forest Plot of Biomass Response (CI ±5)', pad=20, fontsize=14)
plt.tight_layout()

fig.savefig("circular_forest.svg", format='svg')
fig.savefig("circular_forest.pdf", format='pdf')

plt.show()

total = len(df)
print("\n=== Stress factor counts & proportions ===")
for g in groups:
    print(f"{g} : {counts[g]} ({counts[g] / total:.1%})")
