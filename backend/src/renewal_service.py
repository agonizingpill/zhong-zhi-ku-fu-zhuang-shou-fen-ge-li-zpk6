from fastapi import HTTPException
from sqlalchemy import select
from .models import Accession,Lot,GerminationTest,Regeneration,Flowering
from .common import lock,pack,add,reserve,audit
from .vitality import is_legacy,is_stale,needs_recompute,REQUIRED_PROBABILITY,POOLABLE_STATES

def lot_tests(s,lot_id):
    # 锁序约定：本函数只读同批复测行，调用方须先持有种质父行锁。
    return list(s.scalars(select(GerminationTest).where(GerminationTest.lot_id==lot_id).order_by(GerminationTest.id)))

def current_estimate(s,lot_id):
    """同批当前合并估计：必须仍覆盖全部已收录样本，否则视为失效（无当前估计）。"""
    tests=lot_tests(s,lot_id)
    ready=[t for t in tests if t.state=='ready' and t.prediction and not needs_recompute(t.prediction,tests)]
    if not ready:return None,tests
    return max(ready,key=lambda t:t.id),tests

def _gate(prediction):
    probability=prediction.get('probability_below_85')
    if probability is None:return False,'旧估计缺少合并概率依据，请重算后再判断领种门槛'
    if probability<REQUIRED_PROBABILITY:return False,f'合并估计发芽率低于85%的概率为{probability:.1%}，未达到{REQUIRED_PROBABILITY:.0%}领种门槛'
    return True,''

def create_test(s,d,u):
    # 先锁种质父行再锁定种批：补录、领种扣种与worker估计在同一串行点排队。
    pre=s.get(Lot,d.lot_id)
    if not pre:raise HTTPException(404,'种批不存在')
    lock(s,Accession,pre.accession_id)
    lot=lock(s,Lot,d.lot_id)
    snapshot=reserve(s,lot,d.sampled)
    x=GerminationTest(**d.model_dump(),state='recorded',attempt=0);s.add(x);s.flush()
    # 补录即让旧估计失效；重算覆盖新样本前，领种一直暂停。旧预测保留作审计证据。
    # 队列中/计算中的旧尝试也一并作废（尝试号不再匹配），避免其乱序返回写回被补测顶替的样本集。
    for t in lot_tests(s,lot.id):
        if t.id!=x.id and t.state in ('ready','queued','running'):t.state='stale'
    audit(s,u,'抽样发芽复测 '+d.code)
    return pack(x)

def request(s,d,u):
    # 先取种质父锁，再锁定复测行，与create_test/issue保持同一锁序，避免与worker互锁。
    pre=s.get(GerminationTest,d.test_id)
    if not pre:raise HTTPException(404,'复测记录不存在')
    lot=s.get(Lot,pre.lot_id);lock(s,Accession,lot.accession_id)
    test=lock(s,GerminationTest,d.test_id)
    if test.state=='stale':raise HTTPException(409,'同批已有补测收录，旧估计失效，请重算后再申请')
    if test.state!='ready' or not test.prediction:raise HTTPException(409,'先完成当前活力估计；失败评估可重新排队')
    tests=lot_tests(s,lot.id)
    if is_legacy(test.prediction):raise HTTPException(409,'该估计为旧口径，缺少合并概率依据，请重算后再申请')
    if is_stale(test.prediction,tests):raise HTTPException(409,'同批已有补测收录，旧估计失效，请重算后再申请')
    ok,why=_gate(test.prediction)
    if not ok:raise HTTPException(409,why)
    x=Regeneration(**d.model_dump(),lot_id=test.lot_id,vitality_snapshot=test.prediction);audit(s,u,'申请复壮 '+d.code);return add(s,x)

def issue(s,id,u):
    pre=s.get(Regeneration,id)
    if not pre:raise HTTPException(404,'复壮单不存在')
    lot=s.get(Lot,pre.lot_id);lock(s,Accession,lot.accession_id)
    x=lock(s,Regeneration,id)
    if x.state!='requested':raise HTTPException(409,'只能对待领种复壮单扣种，不可重复领用')
    # 补录与领种并发时：先获得父锁的补录先提交，其后的领种读到失效样本集并暂停。
    test,tests=current_estimate(s,lot.id)
    if not test:
        pending=[t.code for t in tests if t.state in POOLABLE_STATES]
        legacy=any(t.state=='ready' and t.prediction and is_legacy(t.prediction) for t in tests)
        detail='该批估计为旧口径，缺少合并概率依据，重算完成前暂停扣种' if legacy else '补测已收录，旧估计失效，重算完成前暂停扣种'
        if pending:detail+='（待重算：'+'、'.join(pending)+'）'
        raise HTTPException(409,detail)
    ok,why=_gate(test.prediction)
    if not ok:raise HTTPException(409,why)
    # 沿用原申请，但把扣种这一刻的依据冻结在复壮单上；之后补测不改已领种单。
    snapshot=reserve(s,lot,x.issued)
    x.issue_snapshot={'reserve':snapshot,'estimate':test.prediction,'estimate_test':pack(test),'sampled_tests':[pack(t) for t in tests],'basis_test_id':test.id}
    x.state='issued';audit(s,u,'主管领种复壮 '+x.code);return pack(x)

def cancel(s,id,u):
    pre=s.get(Regeneration,id)
    if not pre:raise HTTPException(404,'复壮单不存在')
    lot=s.get(Lot,pre.lot_id);lock(s,Accession,lot.accession_id);x=lock(s,Regeneration,id)
    if x.state not in ('requested','issued'):raise HTTPException(409,'仅未下田复壮单可以取消')
    if s.scalar(select(Flowering.id).where(Flowering.regeneration_id==x.id,Flowering.state=='assigned')):raise HTTPException(409,'已分配隔离小区不可直接取消领种')
    if x.state=='issued':lock(s,Lot,lot.id).available+=x.issued
    x.state='cancelled';audit(s,u,'取消并退回未下田种子 '+x.code);return pack(x)
