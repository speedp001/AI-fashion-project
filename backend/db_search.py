# DB에서 전체 데이터셋 조회 파일

import numpy as np

from bson.objectid import ObjectId
from reference import style_array


# search code로 DB조회
def DB_search(client, itemcode):
    fashion = client["fashion"]
    item_coll = fashion[str(itemcode[4:6])]
    style = itemcode[1]
    search_code = itemcode[0] + ".*" + itemcode[2:] + "$"
    # print(search_code)
    singles = item_coll.find({"code": {"$regex": search_code}}, {"_id": 0})

    matched = []
    each_styles = [[] for _ in range(9)]
    result = {}
    result["found"] = None
    result_sets = {}
    result["sets"] = {}

    if len(list(singles)) == 0:
        print("No data found")
        return result

    result["found"] = search_code
    for s in singles:
        each = s["code"][1]
        root = fashion["_cloth_sets"].find_one(
            {"_id": ObjectId(s["root"])}, {"_id": 0})
        if each not in result_sets:
            result_sets[each] = []
        result_sets[each].append(root)
    result["sets"] = result_sets

    return result
