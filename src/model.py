from datetime import datetime, MINYEAR
from sqlalchemy import and_, LargeBinary
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import VARCHAR
from sqlalchemy import CHAR
from sqlalchemy import BINARY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session


Base = declarative_base()

#  Database: ----------------------------------------

class ImdbCelebrity(Base):
    __tablename__ = "imdb_celebrity"

    nconst = Column(CHAR(9), primary_key=True)
    primaryName = Column(VARCHAR(50), nullable=False, default="unknown")

    # gw: cannot be None type, default to year 0,1,1
    # TypeError: SQLite DateTime type only accepts Python datetime and date objects as input.
    birthYear = Column(DateTime)
    deathYear = Column(DateTime)
    primaryProfession = Column(VARCHAR(255))
    knownForTitles = Column(VARCHAR(1024))

    #  parsed resources
    bio = Column(VARCHAR(2048))
    avartar = Column(BINARY)
    avartar_url = Column(VARCHAR(200))
    avartar_blob = Column(LargeBinary)

    # flag to mark number of trials to parse this row
    # combined with an empty field (e.g. primaryName), you can tell whether this row worth further trial
    parse_failure_cnt = Column(Integer, default=0)

    def __init__(self, nconst,
                 primaryName=None,
                 birthYear=None,
                 deathYear=None,
                 primaryProfession=None,
                 knownForTitles=None,

                 bio=None, avartar=None, avartar_url=None,
                 avartar_blob=None,
                 parse_failure_cnt=0
                 ):

        self.nconst = nconst
        self.primaryName=primaryName
        self.birthYear=birthYear
        self.deathYear=deathYear
        self.primaryProfession=primaryProfession
        self.knownForTitles=knownForTitles

        self.bio = bio
        self.avartar = avartar
        self.avartar_url = avartar_url
        self.avartar_blob = avartar_blob
        self.parse_failure_cnt=parse_failure_cnt

    def __repr__(self) -> str:
        return "ImdbCelebrity(%r, %r,%r,%r,%r,%r,%r,%r,)" %(
            self.nconst,
            self.primaryName,
            self.birthYear,
            self.deathYear,
            self.primaryProfession,
            self.knownForTitles,
            self.bio,
            self.avartar_url
        )




class ImdbMsid(Base):
    __tablename__ = "imdb_msid"

    nconst = Column(CHAR(9), primary_key=True)
    msid = Column(VARCHAR(20), nullable=False)

    def __init__(self, nconst, msid):

        self.nconst = nconst
        self.msid = msid

    def __repr__(self) -> str:
        return "ImdbMsid(%r, %r)" %(
            self.nconst,
            self.msid
        )
    
