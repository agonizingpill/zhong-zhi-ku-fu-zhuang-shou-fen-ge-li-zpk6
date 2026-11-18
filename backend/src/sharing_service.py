from fastapi import HTTPException
from sqlalchemy import select
from .models import Lot,Accession,Distribution,GerminationTest
from .common import lock,pack,add,audit,reserve
from .vitality import needs_recompute

def request(s,d,u):
    if not s.get(Lot,d.lot_id):raise HTTPException(404,'种批不存在')
    x=Distribution(**d.model_dump());audit(s,u,'申请种质共享 '+d.code);return add(s,x)
def approve(s,id,u):
    pre=s.get(Distribution,id)
    if not pre:raise HTTPException(404,'共享单不存在')
    lot=s.get(Lot,pre.lot_id);lock(s,Accession,lot.accession_id);d=lock(s,Distribution,id)
    if d.state!='requested':raise HTTPException(409,'共享只能签发一次')
    tests=list(s.scalars(select(GerminationTest).where(GerminationTest.lot_id==lot.id).order_by(GerminationTest.id)))
    # 只认仍对应当前样本集的估计；补录未重算的旧估计不能作为签发依据。
    test=next((t for t in reversed(tests) if t.state=='ready' and t.prediction and not needs_recompute(t.prediction,tests)),None)
    # 共享按该复测本行实测判断，不用同批合并率，避免旧的高发芽结果掩盖新的低发芽结果。
    observed=100*test.germinated/test.sampled if test else None
    if test is None or observed<80:raise HTTPException(409,'共享种批须独立复测发芽率≥80%，后代不可只用继承先验')
    d.reserve_snapshot=reserve(s,lot,d.seeds)|{'test':pack(test)};d.state='approved';audit(s,u,'主管签发种质共享 '+d.code);return pack(d)
def receive(s,id,u):
    d=lock(s,Distribution,id)
    if d.state!='approved':raise HTTPException(409,'只有已签发共享可登记接收，不能重复')
    d.state='received';audit(s,u,'收讫种质共享 '+d.code);return pack(d)
