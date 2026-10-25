import os,io,contextlib,json
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
P='/mnt/data/'
torch.set_num_threads(2)
D=pd.read_csv(P+'upof-model-input-20260512-20260708.csv',parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
a=D.iloc[:-1].reset_index(drop=True);b=D.iloc[1:].reset_index(drop=True)
mask=(a.valid.eq(1)&b.valid.eq(1)&((b.timestamp-a.timestamp)==pd.Timedelta(minutes=15)))
F=pd.DataFrame({'constant':np.ones(len(a)),'theta_minus_T':(a.T_oa_oiko_C+b.T_oa_oiko_C)/2-a.t_bar_C,'electricity':b.elec_kw,'steam':b.steam_kw,'ghi':(a.ghi_kwm2+b.ghi_kwm2)/2,'dhi':(a.dhi_kwm2+b.dhi_kwm2)/2,'dni':(a.dni_kwm2+b.dni_kwm2)/2,'wetbulb_depression':b.T_oa_oiko_C-b.wetbulb_oiko_C,'sin_tod':b.sin_tod,'cos_tod':b.cos_tod,'occupancy':a.occ,'weekend':b.weekend,'q':-b.q_cool_kw}).loc[mask].reset_index(drop=True)
A=a.loc[mask].reset_index(drop=True);B=b.loc[mask].reset_index(drop=True)
X=F.to_numpy();T=A.t_bar_C.to_numpy();truth=B.t_bar_C.to_numpy();y=(truth-T)/.25
train=(B.timestamp<pd.Timestamp('2026-06-25')).to_numpy();orig=np.flatnonzero(mask.to_numpy());vid=np.flatnonzero(~train)
rmse=lambda x,z:float(np.sqrt(np.mean((np.asarray(x)-np.asarray(z))**2)))
reg=LinearRegression(fit_intercept=False).fit(X[train],y[train])
models={'Persistence':None,'Linear regression':reg,'XGBoost':XGBRegressor(max_depth=3,n_estimators=300,learning_rate=.035,subsample=.85,colsample_bytree=.9,n_jobs=2,random_state=42),'Neural network':make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),alpha=.1,early_stopping=True,validation_fraction=.15,max_iter=400,random_state=42,n_iter_no_change=20))}
for name,m in models.items():
 if name not in ('Persistence','Linear regression'):m.fit(X[train],y[train])
def simulation(name,inds,horizon,net=None,sc=None,L=8):
 preds=[];actual=[];p=None;cnt=0;last=None;estimated={}
 with torch.no_grad():
  for i in inds:
   gap=last is None or orig[i]!=orig[last]+1
   if gap or (horizon is not None and cnt>=horizon):p=T[i];cnt=0;estimated={}
   if name=='LSTM':
    inds_seq=np.arange(i-L+1,i+1);rows=X[inds_seq].copy()
    for j,jx in enumerate(inds_seq):rows[j,1]+=T[jx]-estimated.get(int(jx),T[jx])
    rate=float(net(torch.tensor(sc.transform(rows)[None,:,:].astype('float32'))).item())
   elif name=='Persistence':rate=0.
   else:
    row=X[i].copy();row[1]+=T[i]-p
    rate=float(models[name].predict(row.reshape(1,-1))[0])
   estimated[int(i)]=p;p+=.25*rate;preds.append(p);actual.append(truth[i]);cnt+=1;last=i
 return rmse(preds,actual),np.asarray(preds)
rows=[]
for name,m in models.items():
 rates=np.zeros(len(X)) if m is None else m.predict(X)
 row={'model':name,'train_examples':int(train.sum()),'validation_examples':len(vid),'train_one_step_rmse_C':rmse(T[train]+.25*rates[train],truth[train]),'validation_one_step_rmse_C':rmse(T[vid]+.25*rates[vid],truth[vid])}
 for key,h in [('8h',32),('24h',96),('open_loop',None)]:row[key+'_rmse_C']=simulation(name,vid,h)[0]
 rows.append(row)
print('BASELINES',len(X),train.sum(),len(vid),rows[:2], 'C',1/reg.coef_[-1],flush=True)
L=8;eligible=np.array([i for i in range(L-1,len(X)) if orig[i]-orig[i-L+1]==L-1]);etr=train[eligible];ve=eligible[~etr];te=eligible[etr]
sc=StandardScaler().fit(X[train]);xx=sc.transform(X).astype('float32');xseq=torch.tensor(np.stack([xx[i-L+1:i+1] for i in eligible]));yy=torch.tensor(y[eligible].astype('float32'));tri=np.flatnonzero(etr);vali=np.flatnonzero(~etr)
class Model(nn.Module):
 def __init__(self):super().__init__();self.lstm=nn.LSTM(13,16,batch_first=True);self.head=nn.Linear(16,1)
 def forward(self,x):return self.head(self.lstm(x)[0][:,-1,:]).squeeze(-1)
seedrows=[];best_net=None;best_sc=sc;bestseed=None;bestval=1e9
for seed in (42,7,123):
 torch.manual_seed(seed);m=Model();opt=torch.optim.Adam(m.parameters(),lr=.002,weight_decay=.002);loss=nn.MSELoss();rng=torch.Generator().manual_seed(seed)
 best=1e9;bestep=0;state=None
 for epoch in range(110):
  m.train()
  for batch in torch.randperm(len(tri),generator=rng).split(128):
   idx=tri[batch.numpy()];opt.zero_grad();v=loss(m(xseq[idx]),yy[idx]);v.backward();opt.step()
  m.eval()
  with torch.no_grad():pred=m(xseq[vali]).numpy()
  error=rmse(T[ve]+.25*pred,truth[ve])
  if error<best:best=error;bestep=epoch+1;state={k:v.detach().clone() for k,v in m.state_dict().items()}
 m.load_state_dict(state);m.eval()
 r={'seed':seed,'best_epoch':bestep,'validation_one_step_rmse_C':best,'validation_8h_rmse_C':simulation('LSTM',ve,32,m,sc,L)[0],'validation_24h_rmse_C':simulation('LSTM',ve,96,m,sc,L)[0],'validation_open_loop_rmse_C':simulation('LSTM',ve,None,m,sc,L)[0]}
 seedrows.append(r);print('LSTM_SEED',r,flush=True)
 if best<bestval:bestval=best;bestseed=seed;best_net=m;torch.save(state,P+'lstm_best_verified.pt')
with torch.no_grad():rates=best_net(xseq).numpy()
full=eligible
row={'model':'LSTM','train_examples':len(te),'validation_examples':len(ve),'train_one_step_rmse_C':rmse(T[te]+.25*rates[etr],truth[te]),'validation_one_step_rmse_C':rmse(T[ve]+.25*rates[~etr],truth[ve])}
for key,h in [('8h',32),('24h',96),('open_loop',None)]:row[key+'_rmse_C']=simulation('LSTM',ve,h,best_net,sc,L)[0]
rows.append(row)
pd.DataFrame(rows).to_csv(P+'final_verified_results.csv',index=False)
pd.DataFrame(seedrows).to_csv(P+'lstm_seed_robustness.csv',index=False)
common=[]
for name,m in models.items():
 rates=np.zeros(len(ve)) if m is None else m.predict(X[ve]);common.append({'model':name,'examples':len(ve),'one_step_rmse_C':rmse(T[ve]+.25*rates,truth[ve])})
common.append({'model':'LSTM','examples':len(ve),'one_step_rmse_C':row['validation_one_step_rmse_C']})
pd.DataFrame(common).to_csv(P+'final_common_sample.csv',index=False)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
idx=vid[(B.timestamp.iloc[vid]>=pd.Timestamp('2026-06-26'))&(B.timestamp.iloc[vid]<pd.Timestamp('2026-06-29'))]
fig,ax=plt.subplots(figsize=(12,5));ax.plot(B.timestamp.iloc[idx],truth[idx],color='black',lw=2.2,label='Measured')
for name,color in [('Linear regression','#8C98A7'),('XGBoost','#008D8F'),('Neural network','#DE922D')]:
 pp=simulation(name,vid,32)[1];m=np.isin(vid,idx);ax.plot(B.timestamp.iloc[idx],pp[m],label=name,color=color,lw=1.4,alpha=.85)
ppl=simulation('LSTM',ve,32,best_net,sc,L)[1];m=np.isin(ve,idx);ax.plot(B.timestamp.iloc[ve[m]],ppl[m],label='LSTM',color='#8358BE',lw=1.6)
ax.set_title('Measured vs predicted indoor temperature | 8-hour resets | June 26–28');ax.set_ylabel('Indoor temperature (°C)');ax.grid(alpha=.2);ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False);fig.autofmt_xdate();fig.tight_layout();fig.savefig(P+'final_temperature_plot.png',dpi=175,bbox_inches='tight');plt.close(fig)
imp=pd.DataFrame({'feature':F.columns,'importance':models['XGBoost'].feature_importances_}).sort_values('importance',ascending=False);imp.to_csv(P+'final_feature_importance.csv',index=False)
with open(P+'final_project_metadata.json','w') as f:json.dump({'split_training':int(train.sum()),'split_validation':len(vid),'lstm_training_sequences':len(te),'lstm_validation_sequences':len(ve),'best_seed':bestseed,'capacitance':1/reg.coef_[-1],'lstm_seed_selection':'lowest one-step validation error among three seeds; exploratory only','test_window':'not available'},f,indent=2)
print('RESULTS\n',pd.DataFrame(rows).to_string(index=False),'\nFEATURES\n',imp.head(6).to_string(index=False),flush=True)
