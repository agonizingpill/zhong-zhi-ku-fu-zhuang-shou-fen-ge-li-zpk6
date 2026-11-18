import os,time,json,logging
from pathlib import Path
import redis
from sqlalchemy import select
from .db import Session
from .models import Accession,GerminationTest,Lot
from .common import lock
from .vitality import predict

queue=redis.Redis.from_url(os.environ['REDIS_URL'],decode_responses=True)
ARTIFACTS=Path(os.environ.get('ARTIFACTS_DIR','/artifacts'))

def parse(item):
    # 队列载荷为 "<test_id>:<attempt>"；缺少尝试号时回落到库内当前尝试号。
    raw=item[1]
    if ':' in raw:
        test_id,attempt=raw.split(':',1)
        return int(test_id),int(attempt)
    return int(raw),None

def run_test(s,test_id,attempt):
    """一次估计尝试，返回 'skipped' / 'ready' / 'failed'。

    锁序与抽样补录、领种扣种一致：先种质父行，再复测行。先短暂置 running（只锁复测行），
    再在同一父锁下读取当前同批样本并写回，避免与补录/扣种互锁。
    已非 queued、或尝试号已过期时直接放弃：乱序返回的旧消息不会覆盖当前样本；
    重试只在同一复测行推进尝试号，不新增样本行。"""
    test=s.get(GerminationTest,test_id)
    if not test:return 'skipped'
    if test.state!='queued' or test.attempt!=attempt:
        logging.info('skip stale estimate test=%s attempt=%s state=%s',test_id,attempt,test.state);return 'skipped'
    test=lock(s,GerminationTest,test_id)
    if test.state!='queued' or test.attempt!=attempt:
        s.rollback();logging.info('skip stale estimate test=%s attempt=%s state=%s',test_id,attempt,test.state);return 'skipped'
    test.state='running';s.commit()
    try:
        lot=s.get(Lot,test.lot_id)
        lock(s,Accession,lot.accession_id)
        test=lock(s,GerminationTest,test_id)
        if test.state!='running' or test.attempt!=attempt:
            s.rollback();logging.info('estimate superseded test=%s attempt=%s',test_id,attempt);return 'skipped'
        # 与抽样补录、领种扣种共用种质父行锁：读到的同批样本集与写回处于同一串行点。
        siblings=list(s.scalars(select(GerminationTest).where(GerminationTest.lot_id==lot.id).order_by(GerminationTest.id)))
        result=predict(lot,siblings,test,attempt)
        test.prediction=result;test.state='ready';test.error=None
        ARTIFACTS.mkdir(parents=True,exist_ok=True);(ARTIFACTS/f'vitality-{test_id}.json').write_text(json.dumps(result,ensure_ascii=False))
        s.commit();return 'ready'
    except Exception as ex:
        s.rollback();logging.exception('vitality failed')
        try:
            test=lock(s,GerminationTest,test_id)
            # 补录可能已把它顶成 stale；此时不要用失败覆盖新的样本集状态。
            if test.state=='running' and test.attempt==attempt:test.state='failed';test.error=str(ex)[:450]
            s.commit()
        except Exception:
            s.rollback();logging.exception('could not record failure for test=%s',test_id)
        return 'failed'

def main():
    # worker 可能先于 backend 就绪：只做幂等建表/补列，不插入演示数据。
    from .db import Base,engine
    from .seed_service import ensure_columns
    Base.metadata.create_all(engine);ensure_columns()
    while True:
        Path('/tmp/seed-heartbeat').touch()
        try:
            item=queue.blpop('seed-vitality',timeout=2)
            if not item:continue
            id,attempt=parse(item)
            with Session() as s:
                test=s.get(GerminationTest,id)
                if not test:logging.warning('estimate for missing test %s',id);continue
                run_test(s,id,test.attempt if attempt is None else attempt)
        except Exception:logging.exception('worker queue error');time.sleep(1)

if __name__=='__main__':main()
