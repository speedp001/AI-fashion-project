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
    styles = np.zeros(9, dtype=int)
    
    # 성별, 색상, 아이템으로 스타일 목록을 순회하며 db에 일치하는 데이터 순회
    for single in singles:
        
        # 현재 순회 중인 데이터의 스타일을 카운트
        styles[int(single["code"][1])] += 1
        # print(styles) -> [1 1 1 0 1 0 0 0 1]
        
        # 일치한 아이템을 찾으면 matched 딕셔너리에 추가
        if single["code"] == itemcode:
            matched.append(single)
    
    rank = np.argsort(styles)
    # print(rank) -> [3 5 6 7 0 1 2 4 8]
    
    # 해당 조건에 맞는 아이템이 DB에 없을 때
    if len(matched) == 0:
        # DB에 해당 데이터가 없을 때 다른 스타일로 추천
        matched = [style_array[r] for r in rank if styles[r] != 0]
        # print(style_array[rank[0]])
        
        # rank 순위가 높은 순서대로 반환
        return matched, None
    
    
    # 찾은 단품 아이템 정보로 전체 데이터 셋에서 조회
    id = matched[0]["root"]
    sets = fashion["_cloth_sets"].find_one({"_id":ObjectId(id)}, {"_id":0})
    # print(len(sets))
    print(sets)
    for i in sets["items"]:
        if i["item"] == matched[0]["name"]:
            sets["items"].remove(i)
    print("search success")
    return None, sets