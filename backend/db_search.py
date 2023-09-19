####### DB에서 전체 데이터셋 조회 파일

import numpy as np

from bson.objectid import ObjectId
from reference import style_array



####### search code로 DB조회
def DB_search(client, itemcode):    
    # itemcode -> 0(gender) 1(style) 13(color) 16(item) 6자리 'searchcode'
    
    fashion = client["fashion"]
    item_coll = fashion[str(itemcode[4:6])]  # itemcode item에 해당하는 fashion collection
    style = itemcode[1]  # itemcode style
    search_code = "^" + itemcode[0] + ".*" + itemcode[2:] + "$"
    """
    문자열은 item[0]으로 시작해야 합니다.
    문자열의 두 번째 자리는 어떤 문자든지 상관 없습니다 (아무 문자나 와도 됩니다).
    문자열은 "item[2:]" 코드로 끝나야 합니다.
    """
    # print(search_code)
    
    # 해당 코드로 검색
    singles = item_coll.find({"code":{"$regex": search_code}})
    
    # 검색된 목록 딕셔너리
    result = {}  # response로 보낼 dict
    result["found"] = search_code
    result_sets = {}
    result["sets"] = {}
    result["style"] = style

    # cnt를 이용해서 DB 적재 여부 판단
    cnt = 0
    # print(singles[0]["code"])
    
    for single in singles:
        cnt += 1
        each = single["code"][1]
        
        # single root 카테고리로 _cloth_sets 데이터 collection에서 탐색 -> outfit에 저장
        outfit = fashion["_cloth_sets"].find_one({"_id": ObjectId(single["root"])}, {"_id": 0})
        
        if each not in result_sets:
            result_sets[each] = [] #1 -> result_set{"1":[outfit1, outfit2], "2": [outfit1]}
        
        # 출력 결과가 최대 10개까지되도록 설정
        if len(result_sets[each]) < 10:
            result_sets[each].append(outfit)
    
    # print(result_sets) -> [{'set_url': {}... items: [개별 아이템1: {} 개별 아이템2: {}]}]
    result["sets"] = result_sets
    
    # DB에 해당 조건 데이터 없는 경우
    if cnt == 0:
        result["found"] = None

    return result