import os
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase,sessionmaker
url=os.environ['DATABASE_URL']
# 生产 MySQL 固定 READ COMMITTED；SQLite 仅用于本地离线测试，不支持该隔离级别。
engine=create_engine(url,pool_pre_ping=True,**({} if url.startswith('sqlite') else {'isolation_level':'READ COMMITTED'}))
Session=sessionmaker(engine,expire_on_commit=False)
class Base(DeclarativeBase):pass
def session():
    with Session() as s:yield s
