from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()

uri = (
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USERNAME')}:"
    f"{os.getenv('MYSQL_PASSWORD')}@"
    f"{os.getenv('MYSQL_HOST')}:"
    f"{os.getenv('MYSQL_PORT')}/"
    f"{os.getenv('MYSQL_DATABASE')}"
)

engine = create_engine(uri)

with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM Album")).scalar()
    print("Connected successfully. Album count =", count)
