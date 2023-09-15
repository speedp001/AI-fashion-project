from pymongo import MongoClient


def DB_search(client, itemcode):
    fashion = client["fashion"]
    item_coll = fashion[itemcode[4:6]]
    item_coll.findOne({})