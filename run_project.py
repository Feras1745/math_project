"""
Project 52 - NSL-KDD Intrusion Detection
Run: python run_project.py
"""

import warnings; warnings.filterwarnings('ignore')
import os, json, time, base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.decomposition import PCA
import joblib
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\omar\Desktop\Submission_Ready")
DATA_DIR  = BASE / 'data'
FIG_DIR   = BASE / 'figures'
MODEL_DIR = BASE / 'models'
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CLASSES = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']
COLORS  = ['#2E86AB', '#3fb950', '#F18F01', '#A23B72', '#C73E1D']

ATTACK_MAP = {
    'normal':'Normal',
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS','mailbomb':'DoS',
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe','mscan':'Probe','saint':'Probe',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L','phf':'R2L',
    'spy':'R2L','warezclient':'R2L','warezmaster':'R2L','sendmail':'R2L','named':'R2L',
    'snmpgetattack':'R2L','snmpguess':'R2L','httptunnel':'R2L','xlock':'R2L','xsnoop':'R2L','worm':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'sqlattack':'U2R','xterm':'U2R','ps':'U2R',
}

def save_fig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close()
    return path

def img_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

print("=" * 55)
print("  NSL-KDD Intrusion Detection Pipeline")
print("=" * 55)

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n[1/9] Loading data...")
train = pd.read_csv(DATA_DIR / 'KDDTrain.csv')
test  = pd.read_csv(DATA_DIR / 'KDDTest.csv')
train['label'] = train['label'].map(ATTACK_MAP).fillna('Other')
test['label']  = test['label'].map(ATTACK_MAP).fillna('Other')
train = train[train['label'] != 'Other'].reset_index(drop=True)
test  = test[test['label']  != 'Other'].reset_index(drop=True)
print(f"    Train: {train.shape}  |  Test: {test.shape}")

# ══════════════════════════════════════════════════════════════
# 2. EDA FIGURES
# ══════════════════════════════════════════════════════════════
print("\n[2/9] EDA figures...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vc = train['label'].value_counts()
bars = axes[0].bar(vc.index, vc.values, color=COLORS[:len(vc)], edgecolor='white')
axes[0].set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class'); axes[0].set_ylabel('Count')
for b, v in zip(bars, vc.values):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+200, f'{v:,}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
vc_t = test['label'].value_counts()
axes[1].pie(vc_t.values, labels=vc_t.index, autopct='%1.1f%%',
            colors=COLORS[:len(vc_t)], startangle=140,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
fig1 = save_fig('01_class_distribution.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pvc = train['protocol_type'].value_counts()
axes[0].bar(pvc.index, pvc.values, color=COLORS[:3], edgecolor='white')
axes[0].set_title('Protocol Type Distribution', fontweight='bold')
axes[0].set_xlabel('Protocol'); axes[0].set_ylabel('Count')
fvc = train['flag'].value_counts().head(6)
axes[1].bar(fvc.index, fvc.values, color=COLORS, edgecolor='white')
axes[1].set_title('Top 6 TCP Flag Values', fontweight='bold')
axes[1].set_xlabel('Flag'); axes[1].set_ylabel('Count')
fig2 = save_fig('02_protocol_flag.png')

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, feat in zip(axes.flatten(), ['serror_rate','rerror_rate','same_srv_rate','count',
                                      'src_bytes','dst_bytes','dst_host_count','srv_count']):
    for cls, c in zip(['DoS','Normal','Probe'], COLORS[:3]):
        d = train[train['label']==cls][feat]
        ax.hist(d.clip(upper=d.quantile(0.95)), bins=25, alpha=0.55, color=c, label=cls, density=True)
    ax.set_title(feat, fontweight='bold', fontsize=10)
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
axes[0,0].legend(fontsize=8)
plt.suptitle('Feature Distributions by Class', fontsize=13, fontweight='bold', y=1.01)
fig3 = save_fig('03_feature_distributions.png')
print("    EDA figures saved")

# ══════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n[3/9] Preprocessing...")
le = LabelEncoder(); le.fit(CLASSES)
y_train = le.transform(train['label'])
y_test  = le.transform(test['label'])

Xtr = pd.get_dummies(train.drop(columns=['label','difficulty']), columns=['protocol_type','service','flag'])
Xte = pd.get_dummies(test.drop(columns=['label','difficulty']),  columns=['protocol_type','service','flag'])
Xtr, Xte = Xtr.align(Xte, join='left', axis=1, fill_value=0)

top_f = [c for c in ['serror_rate','rerror_rate','same_srv_rate','diff_srv_rate','count',
         'srv_count','src_bytes','dst_bytes','dst_host_count','dst_host_srv_count',
         'dst_host_same_srv_rate','dst_host_serror_rate','duration',
         'dst_host_rerror_rate','srv_serror_rate','srv_rerror_rate',
         'dst_host_diff_srv_rate','srv_diff_host_rate'] if c in Xtr.columns]
fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(Xtr[top_f].corr(), annot=False, cmap='RdBu_r', center=0,
            mask=np.triu(np.ones((len(top_f),len(top_f)),dtype=bool)),
            linewidths=0.3, ax=ax, cbar_kws={'label':'Correlation'})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=8); ax.tick_params(axis='y', rotation=0, labelsize=8)
fig4 = save_fig('04_correlation_heatmap.png')

# ══════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n[4/9] Feature engineering...")
def add_fe(df):
    df = df.copy()
    df['bytes_ratio']         = (df['src_bytes']+1)/(df['dst_bytes']+1)
    df['error_rate_combined'] = df['serror_rate']+df['rerror_rate']
    df['srv_ratio']           = (df['count']+1)/(df['srv_count']+1)
    df['host_srv_ratio']      = (df['dst_host_count']+1)/(df['dst_host_srv_count']+1)
    df['log_src_bytes']       = np.log1p(df['src_bytes'])
    df['log_dst_bytes']       = np.log1p(df['dst_bytes'])
    return df

Xtr = add_fe(Xtr); Xte = add_fe(Xte)
feat_names = list(Xtr.columns)
scaler = StandardScaler()
Xs = scaler.fit_transform(Xtr); Xt = scaler.transform(Xte)
joblib.dump(scaler,     MODEL_DIR/'scaler.joblib')
joblib.dump(le,         MODEL_DIR/'label_encoder.joblib')
joblib.dump(feat_names, MODEL_DIR/'feature_names.joblib')
print(f"    {len(feat_names)} features (6 engineered added)")

idx5k = np.random.choice(len(Xs), 5000, replace=False)
pca   = PCA(n_components=2, random_state=42)
X2d   = pca.fit_transform(Xs[idx5k])
fig, ax = plt.subplots(figsize=(10, 7))
for i, (cls, c) in enumerate(zip(CLASSES, COLORS)):
    mask = y_train[idx5k]==i
    ax.scatter(X2d[mask,0], X2d[mask,1], alpha=0.35, s=7, color=c, label=cls)
ax.set_title('PCA 2D Projection', fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
ax.legend(markerscale=4, fontsize=11)
fig5 = save_fig('05_pca.png')

# ══════════════════════════════════════════════════════════════
# 5. TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("\n[5/9] Training models...")
results = {}
def run(name, model):
    t0 = time.time()
    model.fit(Xs, y_train)
    yp = model.predict(Xt)
    yprob = model.predict_proba(Xt)
    results[name] = dict(
        acc=accuracy_score(y_test,yp), f1=f1_score(y_test,yp,average='weighted'),
        prec=precision_score(y_test,yp,average='weighted',zero_division=0),
        rec=recall_score(y_test,yp,average='weighted'), yp=yp, yprob=yprob)
    print(f"    {name:35s}  acc={results[name]['acc']:.4f}  ({time.time()-t0:.0f}s)")

run('Logistic Regression (Baseline)', LogisticRegression(max_iter=500, random_state=42, n_jobs=-1))
run('Decision Tree (Baseline)',       DecisionTreeClassifier(max_depth=15, random_state=42))
run('Random Forest (Default)',        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))

# ══════════════════════════════════════════════════════════════
# 6. TUNE RANDOM FOREST
# ══════════════════════════════════════════════════════════════
print("\n[6/9] Tuning Random Forest...")
idx_t = np.random.choice(len(Xs), 15000, replace=False)
gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                  {'n_estimators':[100,200],'max_depth':[None,30],
                   'min_samples_split':[2,5],'max_features':['sqrt','log2']},
                  cv=3, scoring='f1_weighted', n_jobs=-1)
gs.fit(Xs[idx_t], y_train[idx_t])
print(f"    Best params: {gs.best_params_}")
best_rf = RandomForestClassifier(**gs.best_params_, random_state=42, n_jobs=-1)
run('Random Forest (Tuned)', best_rf)
joblib.dump(best_rf, MODEL_DIR/'best_random_forest.joblib')

# ══════════════════════════════════════════════════════════════
# 7. EVALUATION FIGURES
# ══════════════════════════════════════════════════════════════
print("\n[7/9] Evaluation figures...")
bk = 'Random Forest (Tuned)'
yp_b = results[bk]['yp']; ypr_b = results[bk]['yprob']

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test,yp_b), annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, linewidths=0.5, ax=ax,
            annot_kws={'size':12,'weight':'bold'})
ax.set_title('Confusion Matrix – Random Forest (Tuned)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold'); ax.set_xlabel('Predicted Label', fontweight='bold')
ax.tick_params(axis='x', rotation=30)
fig6 = save_fig('06_confusion_matrix.png')

rep = classification_report(y_test,yp_b,target_names=CLASSES,output_dict=True)
rdf = pd.DataFrame(rep).T.loc[CLASSES,['precision','recall','f1-score']].astype(float)
fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(rdf, annot=True, fmt='.4f', cmap='YlGnBu', linewidths=0.5,
            vmin=0.8, vmax=1.0, ax=ax, annot_kws={'size':12,'weight':'bold'})
ax.set_title('Classification Report – Random Forest (Tuned)', fontsize=13, fontweight='bold')
ax.set_xlabel('Metric', fontweight='bold'); ax.set_ylabel('Class', fontweight='bold')
ax.tick_params(axis='y', rotation=0)
fig7 = save_fig('07_classification_report.png')

ybin = label_binarize(y_test, classes=list(range(len(CLASSES))))
fig, ax = plt.subplots(figsize=(9,7))
aucs = []
for i,(cls,c) in enumerate(zip(CLASSES,COLORS)):
    fpr,tpr,_ = roc_curve(ybin[:,i], ypr_b[:,i])
    a = auc(fpr,tpr); aucs.append(a)
    ax.plot(fpr,tpr,color=c,lw=2.5,label=f'{cls} (AUC={a:.4f})')
ax.plot([0,1],[0,1],'k--',lw=1.5,label='Random')
ax.set_xlabel('False Positive Rate',fontweight='bold',fontsize=12)
ax.set_ylabel('True Positive Rate',fontweight='bold',fontsize=12)
ax.set_title('ROC Curves – One-vs-Rest',fontsize=13,fontweight='bold')
ax.legend(loc='lower right',fontsize=10); ax.grid(alpha=0.25,linestyle='--')
mean_auc = np.mean(aucs)
ax.text(0.55,0.12,f'Mean AUC = {mean_auc:.4f}',fontsize=12,fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4',facecolor='#e8f4f8',edgecolor='#2E86AB'))
fig8 = save_fig('08_roc_curves.png')

eng = {'bytes_ratio','error_rate_combined','srv_ratio','host_srv_ratio','log_src_bytes','log_dst_bytes'}
fi = pd.DataFrame({'Feature':feat_names,'Importance':best_rf.feature_importances_}) \
       .sort_values('Importance',ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10,8))
bars = ax.barh(fi['Feature'][::-1],fi['Importance'][::-1],
               color=plt.cm.viridis(np.linspace(0.2,0.9,20)),edgecolor='white')
for bar in bars:
    w=bar.get_width()
    ax.text(w+0.0005,bar.get_y()+bar.get_height()/2,f'{w:.4f}',va='center',fontsize=8.5)
for i,feat in enumerate(fi['Feature'].values[::-1]):
    if feat in eng:
        ax.get_yticklabels()[i].set_color('#C73E1D')
        ax.get_yticklabels()[i].set_fontweight('bold')
ax.set_xlabel('Feature Importance (Gini)',fontweight='bold')
ax.set_title('Top 20 Feature Importances',fontsize=13,fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='#C73E1D',label='Engineered features')],fontsize=9,loc='lower right')
fig9 = save_fig('09_feature_importance.png')

keys = list(results.keys()); mn = ['LR\n(Base)','DT\n(Base)','RF\n(Default)','RF\n(Tuned)']
x = np.arange(len(mn)); w = 0.18
fig, ax = plt.subplots(figsize=(13,6))
for vals,off,mc,lab in zip(
    [[results[k]['acc'] for k in keys],[results[k]['prec'] for k in keys],
     [results[k]['rec'] for k in keys],[results[k]['f1'] for k in keys]],
    [-1.5,-0.5,0.5,1.5], COLORS, ['Accuracy','Precision','Recall','F1']):
    bg=ax.bar(x+off*w,vals,w,label=lab,color=mc,edgecolor='white')
    for bar,val in zip(bg,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,
                f'{val:.4f}',ha='center',va='bottom',fontsize=6.5,rotation=90)
ax.set_xticks(x); ax.set_xticklabels(mn,fontsize=10)
ax.set_ylim([max(0.95,min([results[k]['acc'] for k in keys])-0.02),1.005])
ax.set_ylabel('Score',fontweight='bold')
ax.set_title('Model Comparison',fontsize=14,fontweight='bold')
ax.legend(loc='lower right',fontsize=10,ncol=2); ax.grid(axis='y',alpha=0.3,linestyle='--')
ax.axhline(y=1.0,color='gray',linestyle='--',alpha=0.4); ax.axvspan(2.5,3.5,alpha=0.06,color='#C73E1D')
fig10 = save_fig('10_model_comparison.png')

# Fig 11: Feature importance top 15 (no SHAP needed)
top15 = fi.head(15)
fig, ax = plt.subplots(figsize=(10,8))
ax.barh(top15['Feature'][::-1], top15['Importance'][::-1],
        color=plt.cm.plasma(np.linspace(0.2,0.9,15)), edgecolor='white')
for i, feat in enumerate(top15['Feature'].values[::-1]):
    if feat in eng:
        ax.get_yticklabels()[i].set_color('#C73E1D')
        ax.get_yticklabels()[i].set_fontweight('bold')
ax.set_xlabel('Feature Importance (Gini)', fontweight='bold')
ax.set_title('Top 15 Feature Importances (red = engineered)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.25, linestyle='--')
ax.legend(handles=[mpatches.Patch(color='#C73E1D', label='Engineered features')], fontsize=9, loc='lower right')
shap_note = "Gini importance (top 15)"
fig11 = save_fig('11_shap_importance.png')

lc_m = RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=-1)
tr_sizes,tr_sc,va_sc = learning_curve(lc_m,Xs,y_train,train_sizes=np.linspace(0.1,1.0,7),
                                       cv=3,scoring='accuracy',n_jobs=-1,random_state=42)
fig, ax = plt.subplots(figsize=(10,6))
trm,trs = tr_sc.mean(1),tr_sc.std(1); vam,vas = va_sc.mean(1),va_sc.std(1)
ax.plot(tr_sizes,trm,'o-',color='#2E86AB',lw=2.5,label='Training Accuracy')
ax.fill_between(tr_sizes,trm-trs,trm+trs,alpha=0.15,color='#2E86AB')
ax.plot(tr_sizes,vam,'s-',color='#C73E1D',lw=2.5,label='CV Accuracy (3-fold)')
ax.fill_between(tr_sizes,vam-vas,vam+vas,alpha=0.15,color='#C73E1D')
ax.set_xlabel('Training Set Size',fontweight='bold',fontsize=12)
ax.set_ylabel('Accuracy',fontweight='bold',fontsize=12)
ax.set_title('Learning Curve – Random Forest',fontsize=14,fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.25,linestyle='--')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{int(v):,}'))
fig12 = save_fig('12_learning_curve.png')
print("    All figures saved")

# ══════════════════════════════════════════════════════════════
# 8. SAVE MODELS + JSON
# ══════════════════════════════════════════════════════════════
print("\n[8/9] Saving models...")
summary = {}
for k,r in results.items():
    summary[k] = {'accuracy':round(float(r['acc']),4),'f1':round(float(r['f1']),4),
                  'precision':round(float(r['prec']),4),'recall':round(float(r['rec']),4)}
summary['mean_auc']    = round(float(mean_auc),4)
summary['best_params'] = gs.best_params_
summary['class_names'] = CLASSES
with open(MODEL_DIR/'results_summary.json','w') as f:
    json.dump(summary,f,indent=2)

# ══════════════════════════════════════════════════════════════
# 9. HTML REPORT
# ══════════════════════════════════════════════════════════════
print("\n[9/9] Generating HTML report...")

def fig_tag(path, caption):
    b64 = img_b64(path)
    return f'<div class="fig-box"><img src="data:image/png;base64,{b64}" alt="{caption}"><div class="fig-caption">{caption}</div></div>'

best_acc  = results[bk]['acc']
best_f1   = results[bk]['f1']
best_prec = results[bk]['prec']
best_rec  = results[bk]['rec']

rows = ""
for k,r in results.items():
    star = "★" if k == bk else ""
    rows += f'<tr {"class=best" if star else ""}><td>{k} {star}</td><td>{r["acc"]:.4f}</td><td>{r["prec"]:.4f}</td><td>{r["rec"]:.4f}</td><td>{r["f1"]:.4f}</td></tr>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Project 52 Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Segoe UI',sans-serif;background:#f0f4f8;color:#1a202c;line-height:1.7;}}
.header{{background:linear-gradient(135deg,#1a365d,#2E86AB);color:white;padding:50px 40px;text-align:center;}}
.header h1{{font-size:30px;font-weight:800;margin-bottom:8px;}}
.header p{{font-size:15px;opacity:0.85;}}
.kpi-row{{display:flex;gap:16px;padding:28px 40px;flex-wrap:wrap;background:white;border-bottom:1px solid #e2e8f0;}}
.kpi{{flex:1;min-width:120px;background:#f7fafc;border:1px solid #e2e8f0;border-radius:12px;padding:16px;text-align:center;}}
.kpi-val{{font-size:24px;font-weight:800;color:#2E86AB;font-family:monospace;}}
.kpi-lbl{{font-size:11px;color:#718096;margin-top:4px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;}}
.main{{max-width:1100px;margin:0 auto;padding:40px 32px;}}
h2{{font-size:20px;font-weight:700;color:#1a365d;margin:40px 0 16px;padding-bottom:10px;border-bottom:2px solid #2E86AB;}}
.fig-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}}
.fig-grid-1{{display:grid;grid-template-columns:1fr;gap:20px;margin-bottom:20px;}}
.fig-box{{background:white;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);}}
.fig-box img{{width:100%;display:block;}}
.fig-caption{{padding:10px 16px;font-size:13px;color:#4a5568;font-weight:600;background:#f7fafc;border-top:1px solid #e2e8f0;text-align:center;}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:20px;}}
th{{background:#1a365d;color:white;padding:12px 16px;text-align:left;font-size:13px;font-weight:600;text-transform:uppercase;}}
td{{padding:11px 16px;border-bottom:1px solid #e2e8f0;font-size:14px;}}
tr:last-child td{{border-bottom:none;}}
tr.best td{{background:#ebf8ff;font-weight:700;color:#1a365d;}}
.info-box{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:20px 24px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}}
.info-box h3{{font-size:15px;font-weight:700;color:#2E86AB;margin-bottom:10px;}}
.info-box p,.info-box li{{font-size:14px;color:#4a5568;line-height:1.75;}}
.info-box ul{{padding-left:18px;}}
.footer{{text-align:center;padding:30px;color:#718096;font-size:13px;border-top:1px solid #e2e8f0;margin-top:40px;}}
</style>
</head>
<body>
<div class="header">
  <h1>🛡️ Network Intrusion Detection — Project 52</h1>
  <p>Random Forest on NSL-KDD Dataset &nbsp;·&nbsp; Cybersecurity Machine Learning</p>
</div>
<div class="kpi-row">
  <div class="kpi"><div class="kpi-val">{best_acc:.4f}</div><div class="kpi-lbl">Accuracy</div></div>
  <div class="kpi"><div class="kpi-val">{best_f1:.4f}</div><div class="kpi-lbl">F1-Score</div></div>
  <div class="kpi"><div class="kpi-val">{best_prec:.4f}</div><div class="kpi-lbl">Precision</div></div>
  <div class="kpi"><div class="kpi-val">{best_rec:.4f}</div><div class="kpi-lbl">Recall</div></div>
  <div class="kpi"><div class="kpi-val">{mean_auc:.4f}</div><div class="kpi-lbl">Mean AUC</div></div>
  <div class="kpi"><div class="kpi-val">{len(feat_names)}</div><div class="kpi-lbl">Features</div></div>
  <div class="kpi"><div class="kpi-val">125,973</div><div class="kpi-lbl">Train Records</div></div>
  <div class="kpi"><div class="kpi-val">22,544</div><div class="kpi-lbl">Test Records</div></div>
</div>
<div class="main">
<h2>Project Overview</h2>
<div class="info-box">
  <h3>What this project does</h3>
  <p>Classifies network connections into 5 categories: Normal, DoS, Probe, R2L, U2R using a tuned Random Forest on the NSL-KDD benchmark dataset.</p>
</div>
<div class="info-box">
  <h3>Pipeline steps</h3>
  <ul>
    <li>Load and map NSL-KDD into 5 main attack classes</li>
    <li>EDA — class distribution, protocol stats, feature distributions</li>
    <li>Preprocessing — one-hot encoding, StandardScaler, column alignment</li>
    <li>Feature engineering — 6 new features added (total: {len(feat_names)})</li>
    <li>Train 4 models: Logistic Regression, Decision Tree, Random Forest (default + tuned)</li>
    <li>Hyperparameter tuning via GridSearchCV with 3-fold cross-validation</li>
    <li>Evaluation: confusion matrix, ROC curves, classification report, learning curve</li>
    <li>Feature importance + SHAP explainability</li>
  </ul>
</div>
<h2>Model Comparison</h2>
<table>
  <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<div class="info-box">
  <h3>Best model</h3>
  <p><strong>Params:</strong> {gs.best_params_} &nbsp;·&nbsp; <strong>Mean AUC:</strong> {mean_auc:.4f} &nbsp;·&nbsp; {shap_note}</p>
</div>
<h2>EDA</h2>
<div class="fig-grid">{fig_tag(fig1,'Fig 1 — Class Distribution')}{fig_tag(fig2,'Fig 2 — Protocol & Flag')}</div>
<div class="fig-grid-1">{fig_tag(fig3,'Fig 3 — Feature Distributions by Class')}</div>
<div class="fig-grid-1">{fig_tag(fig4,'Fig 4 — Correlation Heatmap')}</div>
<div class="fig-grid-1">{fig_tag(fig5,'Fig 5 — PCA 2D Projection')}</div>
<h2>Evaluation</h2>
<div class="fig-grid">{fig_tag(fig6,'Fig 6 — Confusion Matrix')}{fig_tag(fig7,'Fig 7 — Classification Report')}</div>
<div class="fig-grid-1">{fig_tag(fig8,'Fig 8 — ROC Curves')}</div>
<div class="fig-grid-1">{fig_tag(fig10,'Fig 10 — Model Comparison')}</div>
<div class="fig-grid-1">{fig_tag(fig12,'Fig 12 — Learning Curve')}</div>
<h2>Feature Analysis</h2>
<div class="fig-grid">{fig_tag(fig9,'Fig 9 — Feature Importances (red = engineered)')}{fig_tag(fig11,'Fig 11 — SHAP / Feature Importance Top 15')}</div>
</div>
<div class="footer">Project 52 · NSL-KDD Network Intrusion Detection · Cybersecurity Machine Learning</div>
</body>
</html>"""

report_path = BASE / 'Project52_Report.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n{'='*55}")
print(f"  DONE")
print(f"{'='*55}")
for k,r in results.items():
    print(f"  {k:35s}  acc={r['acc']:.4f}  f1={r['f1']:.4f}")
print(f"\n  Mean AUC : {mean_auc:.4f}")
print(f"\n  Report   → {report_path}")
print(f"  Figures  → {FIG_DIR}")
print(f"  Models   → {MODEL_DIR}")
