"""
Extract demographics from responses.json and produce 4 publication-quality figures.
Saves to:
  images/demo_fig01_gender.png
  images/demo_fig02_age.png
  images/demo_fig03_education.png
  images/demo_fig04_state.png
  images/demo_fig05_combined.png  (2x2 panel)
"""
import json, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# Use the script directory as project base so paths work on any machine.
BASE = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(BASE, "images")
os.makedirs(IMG, exist_ok=True)

# ─── 1. Extract demographics ─────────────────────────────────────────────────
with open(os.path.join(BASE, "responses.json"), encoding="utf-8") as f:
    raw = json.load(f)

top = raw.get("fluency-spam", raw)

ages, edus, states, genders = [], [], [], []

for pid, pdata in top.items():
    if not isinstance(pdata, dict):
        continue
    # demographics can live at top level or inside 'data' entries
    # pattern 1: direct keys
    if "age" in pdata:
        try: ages.append(int(pdata["age"]))
        except: pass
    if "education" in pdata:
        try: edus.append(int(pdata["education"]))
        except: pass
    if "state_ut" in pdata:
        s = str(pdata["state_ut"]).strip()
        if s: states.append(s)
    if "gender" in pdata:
        g = str(pdata["gender"]).strip()
        if g: genders.append(g)
    # pattern 2: inside data list
    for entry in pdata.get("data", []):
        if not isinstance(entry, dict): continue
        if "age" in entry and len(ages) < 200:
            try: ages.append(int(entry["age"]))
            except: pass
        if "education" in entry and len(edus) < 200:
            try: edus.append(int(entry["education"]))
            except: pass
        if "state_ut" in entry and len(states) < 200:
            s = str(entry["state_ut"]).strip()
            if s: states.append(s)
        if "gender" in entry and len(genders) < 200:
            g = str(entry["gender"]).strip()
            if g: genders.append(g)

# Deduplicate by taking one entry per participant (keep unique combos)
# Re-extract cleanly: one record per participant with all 4 fields
records = []
for pid, pdata in top.items():
    if not isinstance(pdata, dict): continue
    rec = {}
    # flat keys
    for key in ("age","education","state_ut","gender"):
        if key in pdata:
            rec[key] = pdata[key]
    # inside data list - find the most complete survey entry
    for entry in pdata.get("data",[]):
        if not isinstance(entry, dict): continue
        for key in ("age","education","state_ut","gender"):
            if key in entry and key not in rec:
                rec[key] = entry[key]
    if rec:
        records.append(rec)

ages    = []
edus    = []
states  = []
genders = []

for r in records:
    try:    ages.append(int(r["age"]))
    except: pass
    try:    edus.append(int(r["education"]))
    except: pass
    if "state_ut" in r and str(r["state_ut"]).strip():
        states.append(str(r["state_ut"]).strip())
    if "gender" in r and str(r["gender"]).strip():
        genders.append(str(r["gender"]).strip())

print(f"Records parsed: {len(records)}")
print(f"  Ages ({len(ages)}):    {sorted(ages)}")
print(f"  Edus ({len(edus)}):    {sorted(edus)}")
print(f"  States ({len(states)}): {Counter(states)}")
print(f"  Genders ({len(genders)}): {Counter(genders)}")

# ─── Colour palette (academic, accessible) ────────────────────────────────────
PALETTE = {
    'blue':   '#4C72B0',
    'orange': '#DD8452',
    'green':  '#55A868',
    'red':    '#C44E52',
    'purple': '#8172B3',
    'brown':  '#937860',
    'pink':   '#DA8BC3',
    'gray':   '#8C8C8C',
    'teal':   '#6AAFBE',
    'lime':   '#C4B800',
}

GENDER_COLORS = {'Male': '#4C72B0', 'Female': '#DD8452',
                 'Non-binary': '#55A868', 'Prefer not to say': '#8C8C8C',
                 'Other': '#8172B3'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
    'savefig.dpi':        150,
    'axes.titlesize':     13,
    'axes.labelsize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
})

# ─── Fig 1: Gender ────────────────────────────────────────────────────────────
gender_counts = Counter(genders)
# normalise labels
g_labels = [k.strip().title() for k in gender_counts.keys()]
g_values = list(gender_counts.values())
g_colors = [GENDER_COLORS.get(l, '#8C8C8C') for l in g_labels]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
# Pie
wedges, texts, autotexts = axes[0].pie(
    g_values, labels=g_labels, colors=g_colors, autopct='%1.0f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops=dict(fontsize=10))
for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
axes[0].set_title('Gender Distribution\n(Pie Chart)', fontweight='bold')

# Bar
bars = axes[1].bar(g_labels, g_values, color=g_colors, edgecolor='white',
                   linewidth=1.2, width=0.55)
axes[1].set_ylabel('Number of Participants')
axes[1].set_title('Gender Distribution\n(Bar Chart)', fontweight='bold')
axes[1].set_ylim(0, max(g_values) * 1.25)
for bar, v in zip(bars, g_values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                 str(v), ha='center', va='bottom', fontweight='bold')

fig.suptitle(f'Participant Gender (N = {len(genders)})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(IMG, 'demo_fig01_gender.png'), bbox_inches='tight')
plt.close()
print("Saved demo_fig01_gender.png")

# ─── Fig 2: Age ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# Histogram
n, bins, patches = axes[0].hist(ages, bins=range(min(ages), max(ages)+2, 1),
                                color=PALETTE['blue'], edgecolor='white',
                                linewidth=0.8, alpha=0.9)
axes[0].axvline(np.mean(ages), color=PALETTE['red'], lw=1.8, ls='--',
                label=f'Mean = {np.mean(ages):.1f}')
axes[0].axvline(np.median(ages), color=PALETTE['green'], lw=1.8, ls='-.',
                label=f'Median = {np.median(ages):.0f}')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Age Histogram', fontweight='bold')
axes[0].legend(fontsize=9)

# Boxplot
bp = axes[1].boxplot(ages, vert=True, patch_artist=True,
                     widths=0.4,
                     boxprops=dict(facecolor=PALETTE['blue'], alpha=0.7),
                     medianprops=dict(color=PALETTE['red'], linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     flierprops=dict(marker='o', color=PALETTE['orange'],
                                     markerfacecolor=PALETTE['orange'],
                                     markersize=6))
# jitter
np.random.seed(42)
jitter = np.random.normal(1, 0.05, len(ages))
axes[1].scatter(jitter, ages, alpha=0.55, s=25, color=PALETTE['teal'],
                zorder=3)
axes[1].set_ylabel('Age (years)')
axes[1].set_xticklabels([''])
axes[1].set_title('Age Distribution\n(Box + Jitter)', fontweight='bold')

stats_text = (f'N={len(ages)}, M={np.mean(ages):.1f}, '
              f'SD={np.std(ages):.1f}, Range={min(ages)}–{max(ages)}')
fig.suptitle(f'Participant Age · {stats_text}',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(IMG, 'demo_fig02_age.png'), bbox_inches='tight')
plt.close()
print("Saved demo_fig02_age.png")

# ─── Fig 3: Education ─────────────────────────────────────────────────────────
edu_counts = Counter(edus)
edu_sorted = sorted(edu_counts.items())
edu_labels = [f'{k} yrs' for k, _ in edu_sorted]
edu_vals   = [v for _, v in edu_sorted]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# Bar
bar_colors = [PALETTE['purple'] if v == max(edu_vals) else PALETTE['blue']
              for v in edu_vals]
bars = axes[0].bar(edu_labels, edu_vals, color=bar_colors, edgecolor='white',
                   linewidth=1.1, width=0.6)
axes[0].set_xlabel('Years of Education')
axes[0].set_ylabel('Number of Participants')
axes[0].set_title('Education Level (Bar Chart)', fontweight='bold')
for bar, v in zip(bars, edu_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.08,
                 str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

# KDE / Step histogram
axes[1].hist(edus, bins=range(min(edus), max(edus)+2, 1),
             color=PALETTE['purple'], edgecolor='white',
             linewidth=0.8, alpha=0.85, label='Frequency')
axes[1].axvline(np.mean(edus), color=PALETTE['red'], lw=2, ls='--',
                label=f'Mean = {np.mean(edus):.1f} yrs')
axes[1].axvline(np.median(edus), color=PALETTE['green'], lw=2, ls='-.',
                label=f'Median = {np.median(edus):.0f} yrs')
axes[1].set_xlabel('Years of Education')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Education Distribution\n(Histogram)', fontweight='bold')
axes[1].legend(fontsize=9)

fig.suptitle(f'Participant Education Level (N = {len(edus)}) · '
             f'M = {np.mean(edus):.1f} yrs, SD = {np.std(edus):.1f}',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(IMG, 'demo_fig03_education.png'), bbox_inches='tight')
plt.close()
print("Saved demo_fig03_education.png")

# ─── Fig 4: State ─────────────────────────────────────────────────────────────
state_counts = Counter(states)
state_sorted = state_counts.most_common()
s_labels = [s for s, _ in state_sorted]
s_vals   = [v for _, v in state_sorted]

# Colour: North zone vs South zone heuristic
north_states = {'Uttar Pradesh','Bihar','Rajasthan','Madhya Pradesh',
                'Chhattisgarh','Jharkhand','Uttarakhand','Haryana',
                'Punjab','Himachal Pradesh','Delhi','Jammu and Kashmir'}
s_colors = [PALETTE['blue'] if s in north_states else PALETTE['orange']
            for s in s_labels]

fig, ax = plt.subplots(figsize=(10, max(4, len(s_labels)*0.45 + 1)))
y_pos = range(len(s_labels))
bars = ax.barh(list(y_pos), s_vals, color=s_colors,
               edgecolor='white', linewidth=0.8, height=0.65)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(s_labels, fontsize=9)
ax.set_xlabel('Number of Participants', fontsize=11)
ax.set_title(f'Participant State / UT Distribution (N = {len(states)})',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()
for bar, v in zip(bars, s_vals):
    ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2.,
            str(v), va='center', fontsize=9, fontweight='bold')
# Legend
north_patch = mpatches.Patch(color=PALETTE['blue'],  label='North/Central zone')
south_patch = mpatches.Patch(color=PALETTE['orange'], label='South/East zone')
ax.legend(handles=[north_patch, south_patch], frameon=False, fontsize=9,
          loc='lower right')
ax.set_xlim(0, max(s_vals) * 1.18)
plt.tight_layout()
fig.savefig(os.path.join(IMG, 'demo_fig04_state.png'), bbox_inches='tight')
plt.close()
print("Saved demo_fig04_state.png")

# ─── Fig 5: Combined 2×2 demographics panel ──────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Participant Demographics Overview', fontsize=16,
             fontweight='bold', y=0.98)

# ── panel A: Gender pie ──
ax_g = fig.add_subplot(2, 2, 1)
wedges, texts, autotexts = ax_g.pie(
    g_values, labels=g_labels, colors=g_colors, autopct='%1.0f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1.5))
for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
ax_g.set_title(f'(A) Gender (N={len(genders)})', fontweight='bold', fontsize=12)

# ── panel B: Age histogram ──
ax_a = fig.add_subplot(2, 2, 2)
ax_a.hist(ages, bins=range(min(ages), max(ages)+2, 1),
          color=PALETTE['teal'], edgecolor='white', alpha=0.9)
ax_a.axvline(np.mean(ages), color=PALETTE['red'], lw=2, ls='--',
             label=f'Mean={np.mean(ages):.1f}')
ax_a.axvline(np.median(ages), color=PALETTE['green'], lw=2, ls='-.',
             label=f'Median={np.median(ages):.0f}')
ax_a.set_xlabel('Age (years)')
ax_a.set_ylabel('Frequency')
ax_a.set_title(f'(B) Age (N={len(ages)})', fontweight='bold', fontsize=12)
ax_a.legend(fontsize=9)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# ── panel C: Education bar ──
ax_e = fig.add_subplot(2, 2, 3)
ax_e.bar(edu_labels, edu_vals,
         color=[PALETTE['purple'] if v==max(edu_vals) else '#9380C4' for v in edu_vals],
         edgecolor='white', linewidth=0.8)
ax_e.set_xlabel('Years of Education')
ax_e.set_ylabel('Count')
ax_e.set_title(f'(C) Education (N={len(edus)})', fontweight='bold', fontsize=12)
ax_e.tick_params(axis='x', rotation=30)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)

# ── panel D: State horizontal bar ──
ax_s = fig.add_subplot(2, 2, 4)
yp = range(len(s_labels))
ax_s.barh(list(yp), s_vals, color=s_colors, edgecolor='white',
          linewidth=0.6, height=0.6)
ax_s.set_yticks(list(yp))
ax_s.set_yticklabels(s_labels, fontsize=8)
ax_s.set_xlabel('Count')
ax_s.set_title(f'(D) State/UT (N={len(states)})', fontweight='bold', fontsize=12)
ax_s.invert_yaxis()
ax_s.spines['top'].set_visible(False)
ax_s.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(IMG, 'demo_fig05_combined.png'), bbox_inches='tight')
plt.close()
print("Saved demo_fig05_combined.png")
print(f"All demographic figures saved to: {IMG}")
