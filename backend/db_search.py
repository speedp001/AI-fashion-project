# DB에서 전체 데이터셋 조회 파일

import numpy as np

from bson.objectid import ObjectId
from reference import style_array



####### search code로 DB조회
def DB_search(client, itemcode):    
    # itemcode -> 0(gender) 1(style) 13(color) 16(item) 6자리 'searchcode'
    
    fashion = client["fashion"]
    item_coll = fashion[str(itemcode[4:6])]  # itemcode item에 해당하는 fashion collection
    style = itemcode[1]  # itemcode style
    search_code = itemcode[0] + ".*" + itemcode[2:] + "$"
    """
    문자열은 item[0]으로 시작해야 합니다.
    문자열의 두 번째 자리는 어떤 문자든지 상관 없습니다 (아무 문자나 와도 됩니다).
    문자열은 "item[2:]" 코드로 끝나야 합니다.
    """
    # print(search_code)
    
    # 해당 코드로 검색
    singles = item_coll.find({"code":{"$regex": search_code}}, {"_id":0})
    # {"_id": 0}는 결과에서 _id 필드를 제외하도록 지정
    
    # 검색된 목록 딕셔너리
    matched = []
    each_styles = [[] for _ in range(9)]
    result = {}
    result["found"] = search_code
    result_sets = {}
    result["sets"] = {}
    result["style"] = style

    result["found"] = search_code
    cnt = 0  # mongodb에서 found한 cursors들을 한번 iterate 하면 처음으로 다시 돌아갈 수 없음 ㅠ
    # print(singles[0]["code"])
    for s in singles:
        cnt += 1
        each = s["code"][1]
        root = fashion["_cloth_sets"].find_one(
            {"_id": ObjectId(s["root"])}, {"_id": 0})
        if each not in result_sets:
            result_sets[each] = []
        result_sets[each].append(root)
    result["sets"] = result_sets
    if cnt == 0:
        result["found"] = None

    return result
