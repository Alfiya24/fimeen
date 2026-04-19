import certifi
from pymongo import MongoClient
from src.config import MONGODB_URI

client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
print(client.list_database_names())