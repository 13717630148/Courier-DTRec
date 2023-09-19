import os
import pickle
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

DIS_CLUSTER = 5 
LOC_T_PREV = 180
LOC_T_POST  = 3600
LOC_NEAR_STAY = 50
LOC_NEAR_STAY2 = LOC_NEAR_STAY ** 2
LOC_NEAR_STAYSQRT2 = LOC_NEAR_STAY * 2 ** 0.5


def cluster_algo(ps):
    db = DBSCAN(eps=DIS_CLUSTER, min_samples=1, n_jobs=-1).fit(ps)
    labels = db.labels_
    return labels


def correct_order_address(wave_data, stay_ref, use_cache=True):
    date2waves = defaultdict(list)
    for w in wave_data:
        date2waves[w["date"]].append(w)

    stays_all = [s for x in wave_data for s in x["stays"]]
    if stay_ref:
        cache_path = "data/locids.pkl"
    else:
        cache_path = "data/locids_no_stay_ref.pkl"
    if use_cache and os.path.exists(cache_path):
        print("use cache", cache_path)
        locids = pickle.load(open(cache_path, "rb"))
    else:
        locs_stay = [s["point"][:2] for s in stays_all]
        locids = cluster_algo(locs_stay)
        pprint(sorted(list(Counter(locids).items()), key=lambda x: -x[1])[:20])
        pickle.dump(locids, open(cache_path, "wb"))
    locid2xy = defaultdict(list)
    for s, locid in zip(stays_all, locids):
        s["locid"] = locid
        assert locid != -1
        locid2xy[locid].append(s["point"][:2])
    locid2xy = {locid: np.mean(np.array(xys), axis=0) for locid, xys in locid2xy.items()}

    def get_wave_occurance(orders, stays):
        stays = [(s["trange"], s["locid"]) for s in stays]
        adsid2locids = defaultdict(set)
        bid2locids = defaultdict(set)
        adsid2locids_strong = defaultdict(set)
        bid2locids_strong = defaultdict(set)
        all_adsid = set()
        all_bid = set()
        for o in orders:
            t = o["finish_time"]
            all_adsid.add(o["address_id"])
            all_bid.add(o["building_id"])
            for (t1, t2), locid in stays:
                if t1 - LOC_T_PREV <= t and t <= t2 + LOC_T_POST:
                    adsid2locids[o["address_id"]].add(locid)
                    bid2locids[o["building_id"]].add(locid)
                    if t1 <= t <= t2 + LOC_T_PREV:
                        adsid2locids_strong[o["address_id"]].add(locid)
                        bid2locids_strong[o["building_id"]].add(locid)
        bid2locids.pop(-1, None)
        bid2locids_strong.pop(-1, None)
        all_bid.discard(-1)
        return all_adsid, all_bid, adsid2locids, bid2locids, adsid2locids_strong, bid2locids_strong
    
    if stay_ref:
        cache_path = "data/occs.pkl"
    else:
        cache_path = "data/occs_no_stay_ref.pkl"
    if use_cache and os.path.exists(cache_path):
        print("use cache", cache_path)
        date2occs = pickle.load(open(cache_path, "rb"))
    else:
        date2occs = {
            date: [get_wave_occurance(w["orders"], w["stays"]) for w in waves]
            for date, waves in date2waves.items()
        }
        pickle.dump(date2occs, open(cache_path, "wb"))

    def find_near_locs(x, y):
        nonlocal locid2xy
        near_locs = set()
        for locid, (x1, y1) in locid2xy.items():
            a = abs(x - x1)
            if a < LOC_NEAR_STAYSQRT2:
                b = abs(y - y1)
                if a + b < LOC_NEAR_STAYSQRT2:
                    if a ** 2 + b ** 2 < LOC_NEAR_STAY2:
                        near_locs.add(locid)
        return near_locs

    def propose_locs(adsid, bid, locids, occs):
        if not locids:
            return []

        locid2occads = defaultdict(int)
        locid2occads_strong = defaultdict(int)
        occs_ads = [x for x in occs if adsid in x[0]]
        for locid in locids:
            for occ in occs_ads:
                locid2occads[locid] += locid in occ[2][adsid]  # adsid2locids
                locid2occads_strong[locid] += locid in occ[4][adsid]
        locid2score_ads = {
            i: (locid2occads[i] + locid2occads_strong[i]) / (len(occs_ads) + 1e-12)
            for i in locids
        }
        if bid != -1:
            locid2occbd = defaultdict(int)
            locid2occbd_strong = defaultdict(int)
            occs_bd = [x for x in occs if bid in x[1]]
            for locid in locids:
                for occ in occs_bd:
                    locid2occbd[locid] += locid in occ[3][adsid]  # bid2locids
                    locid2occbd_strong[locid] += locid in occ[5][adsid]
            locid2score_bd = {
                i: (locid2occbd[i] + locid2occbd_strong[i]) / (len(occs_bd) + 1e-12)
                for i in locids
            }
        else:
            locid2score_bd = {i: 0 for i in locids}

        locids = {i for i in locids if locid2score_ads[i] > 0.2 or locid2score_bd[i] > 0.2}
        if not locids:
            return []

        if bid != -1:
            occs_nobd = [x for x in occs if bid not in x[1]]
            locid2occnobd = defaultdict(int)
            for locid in locids:
                for occ in occs_nobd:
                    for v in occ[2].values():  # adsid2locids
                        if locid in v:
                            locid2occnobd[locid] += 1
                            break
            locid2score_nobd = {
                i: locid2occnobd[i]/ (len(occs_nobd) + 1e-12)
                for i in locids
            }
        else:
            locid2score_nobd = {i: 0 for i in locids}

        locid2score = {
            i: 
            locid2score_ads[i] + 0.5 * locid2score_bd[i] - 0.5 * locid2score_nobd[i] 
            for i in locids
        }
        locid_score = [(l, s) for l, s in locid2score.items() if s > 0.5]
        locid_score.sort(key=lambda x: -x[1])
        return locid_score[:3]

    if stay_ref:
        cache_path = "data/xy2near_locs.pkl"
    else:
        cache_path = "data/xy2near_locs_no_stay_ref.pkl"
    if use_cache and os.path.exists(cache_path):
        print("use cache", cache_path)
        xy2near_locs = pickle.load(open(cache_path, "rb"))
    else:
        xy2near_locs = {}

    for date, waves in tqdm(date2waves.items()):
        occs = [occ for d, occs in date2occs.items() for occ in occs if d != date]
        for w in waves:
            stays = [(s["trange"], s["point"][:2]) for s in w["stays"]]
            for o in w["orders"]:
                t = o["finish_time"]
                locids = set()
                for (t1, t2), xy in stays:
                    if t1 - LOC_T_PREV <= t and t <= t2 + LOC_T_POST:
                        if xy in xy2near_locs:
                            locids |= xy2near_locs[xy]
                        else:
                            near_locs = find_near_locs(*xy)
                            locids |= near_locs
                            xy2near_locs[xy] = near_locs
                locid_scores = propose_locs(o["address_id"], o["building_id"], locids, occs)
                o["loc_scores"] = [(locid2xy[locid], score) for locid, score in locid_scores]

    pickle.dump(xy2near_locs, open(cache_path, "wb"))

    return wave_data


if __name__ == "__main__":
    wave_data = pickle.load(open("dataset/wave_data_no_stay_ref.pkl", "rb"))
    wave_data = correct_order_address(wave_data, stay_ref=False, use_cache=True)
    pickle.dump(wave_data, open("dataset/wave_data_no_stay_ref_corrected.pkl", "wb"))

    wave_data = pickle.load(open("dataset/wave_data.pkl", "rb"))
    wave_data = correct_order_address(wave_data, stay_ref=True, use_cache=True)
    pickle.dump(wave_data, open("dataset/wave_data_corrected.pkl", "wb"))