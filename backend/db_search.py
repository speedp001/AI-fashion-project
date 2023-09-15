from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
from reference import style_dict


def DB_search(client, itemcode):
    fashion = client["fashion"]
    item_coll = fashion[str(itemcode[4:6])]
    style = itemcode[1]
    search_code = itemcode[0] + ".*" + itemcode[2:] + "$"
    # print(search_code)
    singles = item_coll.find({"code":{"$regex": search_code}}, {"_id":0})
    matched = []
    styles = np.zeros(9, dtype=int)
    
    for s in singles:
        styles[int(s["code"][1])] += 1
        if s["code"] == itemcode:
            matched.append(s)
            
    rank = np.argsort(styles[:-1])
    
    if len(matched) == 0:
        matched = [style_dict[r] for r in rank if styles[r] != 0]
        print(style_dict[rank[0]])
        return matched, None
    id = matched[0]["root"]
    sets = fashion["_cloth_sets"].find_one({"_id":ObjectId(id)}, {"_id":0})
    # print(len(sets))
    print(sets)
    print("success")
    return None, sets