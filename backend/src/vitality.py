import hashlib,json,math,random,time
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 低活力阈值与领种所需的后验概率门槛。
LOW_RATE=85.0
REQUIRED_PROBABILITY=0.8
# 尚无可用当前估计、可排队重算的样本状态：新录、排队、计算中、失败，以及被补测顶成失效的旧估计。
POOLABLE_STATES=('recorded','queued','running','failed','stale')

def training_rows():
    rng=random.Random(32);rows=[]
    for i in range(180):
        months=rng.uniform(0,60);temp=rng.uniform(-18,22);initial=rng.uniform(80,98)
        actual=max(0,min(100,initial-.29*months-.34*(temp+18)+rng.gauss(0,2.2)))
        rows.append([round(months,4),round(temp,4),round(initial,4),round(actual,4)])
    return rows

def sample_key(tests):
    """同批样本签名：每个复测行只出现一次，重试/重算不改变签名，新增补测才改变。"""
    def val(x,k):return x[k] if isinstance(x,dict) else getattr(x,k)
    return '|'.join(f'{val(t,"id")}:{val(t,"sampled")}:{val(t,"germinated")}' for t in sorted(tests,key=lambda t:val(t,'id')))

def is_stale(prediction,tests):
    """估计是否仍对应当前同批样本集；补录后旧估计即失效。"""
    basis=(prediction or {}).get('basis') or {}
    return basis.get('sample_key')!=sample_key(tests)

def is_legacy(prediction):
    """旧口径估计（无合并样本依据）只能重算，不能直接用于领种门槛。"""
    return not ((prediction or {}).get('basis') or {}).get('sample_key')

def needs_recompute(prediction,tests):
    return is_legacy(prediction) or is_stale(prediction,tests)

def _cdf(z):return .5*(1+math.erf(z/math.sqrt(2)))

def predict(lot,tests,carrier,attempt=0):
    rows=training_rows();data=np.array(rows);train,hold=train_test_split(np.arange(len(rows)),test_size=.25,random_state=32)
    scaler=StandardScaler();x=scaler.fit_transform(data[train,:3]);model=BayesianRidge();model.fit(x,data[train,3])
    mean,std=model.predict(scaler.transform(data[hold,:3]),return_std=True)
    value,sigma=model.predict(scaler.transform([[lot.stored_months,lot.temperature,lot.initial_rate]]),return_std=True)
    prior_mean=float(value[0]);prior_sd=max(float(sigma[0]),1e-6)

    # 同批全部实测共同参与合并；每个复测行只计一次，重试不会把同一份计数再加一遍。
    ordered=sorted(tests,key=lambda t:t.id)
    observations=[{'test_id':t.id,'code':t.code,'sampled':t.sampled,'germinated':t.germinated,'observed_pct':round(100*t.germinated/t.sampled,3),'state':'ready' if t.id==carrier.id else t.state,'carrier':t.id==carrier.id} for t in ordered]
    pooled_sampled=sum(t.sampled for t in ordered);pooled_germinated=sum(t.germinated for t in ordered)
    raw_observed=100*pooled_germinated/pooled_sampled
    # 比例的连续性校正，避免全发芽/全不发芽时标准误为0。
    adjusted=(pooled_germinated+.5)/(pooled_sampled+1)
    observed_sd=100*math.sqrt(max(adjusted*(1-adjusted),0)/pooled_sampled)
    observed_sd=max(observed_sd,.5)

    # 正态—正态合并：模型预测分布为批真实发芽率先验，同批实测为二项似然的正态近似。
    prior_weight=1/prior_sd**2;observed_weight=1/observed_sd**2
    posterior_sd=1/math.sqrt(prior_weight+observed_weight)
    posterior_mean=posterior_sd**2*(prior_mean*prior_weight+100*adjusted*observed_weight)
    probability_below=_cdf((LOW_RATE-posterior_mean)/posterior_sd)

    return {
        'estimate_pct':round(posterior_mean,3),
        'lower95_pct':round(max(0,posterior_mean-1.96*posterior_sd),3),
        'upper95_pct':round(min(100,posterior_mean+1.96*posterior_sd),3),
        'observed_pct':round(raw_observed,3),
        'model_mean_pct':round(prior_mean,3),
        'model_sd_pct':round(prior_sd,3),
        'model_lower95_pct':round(max(0,prior_mean-1.96*prior_sd),3),
        'model_upper95_pct':round(min(100,prior_mean+1.96*prior_sd),3),
        'pooled_sampled':pooled_sampled,
        'pooled_germinated':pooled_germinated,
        'pooled_observed_pct':round(raw_observed,3),
        'pooled_observed_adjusted_pct':round(100*adjusted,3),
        'observed_sd_pct':round(observed_sd,3),
        'posterior_mean_pct':round(posterior_mean,3),
        'posterior_sd_pct':round(posterior_sd,3),
        'posterior_lower95_pct':round(max(0,posterior_mean-1.96*posterior_sd),3),
        'posterior_upper95_pct':round(min(100,posterior_mean+1.96*posterior_sd),3),
        'probability_below_85':round(probability_below,4),
        'threshold_pct':LOW_RATE,
        'required_probability':REQUIRED_PROBABILITY,
        'renewal_required':probability_below>=REQUIRED_PROBABILITY,
        'observations':observations,
        'pooling':'Normal prior from storage model (mean, predictive sd) fused with binomial likelihood of all same-lot tests, continuity-adjusted; normal approximation on the percentage scale',
        'sample_count_rule':'Each germination test row contributes once; re-estimating a failed test reuses the same counts',
        'holdout_mae_percentage_points':round(float(mean_absolute_error(data[hold,3],mean)),4),
        'holdout_interval95_coverage':round(float(np.mean((data[hold,3]>=mean-1.96*std)&(data[hold,3]<=mean+1.96*std))),4),
        'train_rows':len(train),'holdout_rows':len(hold),'train_indices':train.tolist(),'holdout_indices':hold.tolist(),'seed':32,
        'source_checksum':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),
        'input_snapshot':{'lot_id':lot.id,'stored_months':lot.stored_months,'temperature':lot.temperature,'initial_rate':lot.initial_rate,'carrier_test_id':carrier.id,'carrier_sampled':carrier.sampled,'carrier_germinated':carrier.germinated},
        'basis':{'sample_key':sample_key(ordered),'test_ids':[t.id for t in ordered],'carrier_test_id':carrier.id,'attempt':attempt,'computed_at':round(time.time(),3)},
        'model':'StandardScaler(train only) + BayesianRidge',
        'interval':'Gaussian posterior mean ±1.96 posterior sd, clipped to 0–100',
    }
