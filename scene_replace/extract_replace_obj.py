import json
import random
import os
import argparse

from pxr import Usd, UsdGeom, Gf

from utils import get_world_bbox


def get_similar_size_usd(target_root, size, threshold=0.75):
    best_usd_path = None
    best_rate = -1
    for sub_dir in os.listdir(target_root):
        js_file = os.path.join(target_root, sub_dir, "base_info.json")
        base_info = json.load(open(js_file))
        usd_path = base_info["usd_path"]
        new_size = base_info["size"]
        if abs(new_size[0]) > 1e6:
            continue
        ratios = [min(size[i], new_size[i]) / max(size[i], new_size[i]) for i in range(3)]
        rate = sum(ratios) / 3  # 越接近 1 越好
        if rate >= threshold:  # decrease search time
            best_usd_path = usd_path
            if random.random() > 0.5:
                break
        if best_rate < rate:
            best_rate = rate
            best_usd_path = usd_path

    return best_usd_path


def find_new_usd_path(old_key, new_usd_json_dic, size):
    for key1 in new_usd_json_dic.keys():
        for key2 in new_usd_json_dic[key1].keys():
            if old_key in new_usd_json_dic[key1][key2].keys():
                target_key1 = None
                for key3 in new_usd_json_dic[key1][key2].keys():
                    if old_key in key3:
                        target_key1 = key3
                        break

                target_root = new_usd_json_dic[key1][key2][target_key1]
                return get_similar_size_usd(target_root, size)
    return None


def get_candidate_categories(new_usd_json_dic):
    candi_cates = set()
    for key1 in new_usd_json_dic.keys():
        for key2 in new_usd_json_dic[key1].keys():
            for key3 in new_usd_json_dic[key1][key2].keys():
                candi_cates.add(key3.lower())
    return candi_cates


def get_replace_objs(stage, new_usd_json_dic):
    import gensim.downloader as api
    import numpy as np
    from wordsegment import load, segment
    load()
    model = api.load("glove-wiki-gigaword-50")

    def phrase_vector(phrase, model):
        words = phrase.lower().split()
        vecs = [model[w] for w in words if w in model]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    candi_cates = get_candidate_categories(new_usd_json_dic)
    cate_vecs = {}
    for cate in candi_cates:
        split_cate = cate.split("_")
        items = []
        for item in split_cate:
            item = segment(item)
            items.extend(item)
        new_cate = " ".join(items)
        cate_vecs[cate] = phrase_vector(new_cate, model)

    old_prim_path_dict = {}
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName().lower()
        if "bbox" not in prim_name:
            continue
        size, _, _ = get_world_bbox(stage, prim)
        cate_name = " ".join(prim_name.split("_")[1:-1])  # example: bbox_dinning_table_1
        q_vec = phrase_vector(cate_name, model)
        best_cate = None
        best_sim = -1
        for cate, cate_vec in cate_vecs.items():
            cos_sim = np.dot(q_vec, cate_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(cate_vec))
            if cos_sim > best_sim:
                best_sim = cos_sim
                best_cate = cate
        if "wine cabinet" in cate_name:
            best_cate = "cabinet"
        if best_cate in old_prim_path_dict:
            key = list(old_prim_path_dict[best_cate].keys())[0]
            new_usd_path = old_prim_path_dict[best_cate][key]
            old_prim_path_dict[best_cate][prim_path] = new_usd_path
            continue
        new_usd_path = find_new_usd_path(best_cate, new_usd_json_dic, size)
        if new_usd_path is None:
            new_usd_path = ""
        if best_cate in old_prim_path_dict:
            old_prim_path_dict[best_cate][prim_path] = new_usd_path
        else:
            old_prim_path_dict[best_cate] = {prim_path: new_usd_path}

    return old_prim_path_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert USD materials with assets")
    parser.add_argument("--usd_path", type=str, required=True,
                        help="usd file needed to be replaced.")

    args = parser.parse_args()
    usd_path = args.usd_path
    new_usd_json_dic = json.load(open("./home_scenes.json"))
    stage = Usd.Stage.Open(usd_path)
    prim_match_dict = get_replace_objs(stage, new_usd_json_dic)
    json.dump(prim_match_dict, open("./prim_match_dict.json", "w"), indent=4)
