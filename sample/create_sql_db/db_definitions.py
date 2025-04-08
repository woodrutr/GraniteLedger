from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


class TESTBatteryEfficiency(Base):
    __tablename__ = 'TESTBatteryEfficiency'
    id = Column(Integer, primary_key=True)
    tech = Column(Integer)
    BatteryEfficiency = Column(Float)


class TESTHourstoBuy(Base):
    __tablename__ = 'TESTHourstoBuy'
    id = Column(Integer, primary_key=True)
    tech = Column(Integer)
    HourstoBuy = Column(Float)
