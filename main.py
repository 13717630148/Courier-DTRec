import pickle
import random
from collections import defaultdict
from copy import deepcopy
from itertools import groupby

import numpy as np
import yaml
from shapely.geometry import Point
from sklearn import linear_model
from tqdm import tqdm

from coarse_rec import cal_score_s, cal_score_t, coarse_rec
from constants import *
from fine_rec import (DeepFineRec, fine_rec_abm, fine_rec_mid, fine_rec_smt,
                      fine_rec_unf)

MIN_MATCH_SCORE = 0.1
BIG_MATCH_SCORE = 0.7

TMIN_BASE = 10
TMAX_BASE = 180
TMIN_CUNIT = 5
TMAX_CUNIT = 20
TMIN_ORDER = 3
TMAX_ORDER = 30
TMIN_STAIR = 12
TMAX_STAIR = 40 
TMIN_ELE_WAIT1 = 3
TMAX_ELE_WAIT1 = 20   
TMIN_ELE_WAIT2 = 3
TMAX_ELE_WAIT2 = 10
TMIN_ELE = 4
TMAX_ELE = 8

TRAIN_DAYS = list(range(801, 832)) + list(range(901, 906))

random.seed(233)

N_ITER_IN = 5
N_ITER_OUT = 2


def match_s_o(s, o):
    adr_ref, method = config["adr_ref"], config["coarse"]
    score_t = 1 if method == "s" else cal_score_t(*s["trange"], o["finish_time"])
    if score_t < MIN_MATCH_SCORE:
        return 0, None
    if not adr_ref:
        o["xys"] = [o["xy"]]
    else:
        if o["building_id"] != -1:
            xys = [o["xy"]] + [buildings[o["building_id"]]["gate_xy"]] + [x[0] for x in o["loc_scores"]]
        else:
            xys = [o["xy"]] + [x[0] for x in o["loc_scores"]]
        if o["address_xy"]:
            xys.append(o["address_xy"])
        o["xys"] = xys
    matched_results = []
    for xy in o["xys"]:
        score = score_t * (1 if method == "t" else cal_score_s(*s["point"][:2], *xy))
        if score > BIG_MATCH_SCORE:
            return score, xy
        elif score >= MIN_MATCH_SCORE:
            matched_results.append([xy, score])
    if matched_results:
        max_score = max(x[-1] for x in matched_results)
        tmp = [x for x in matched_results if x[-1] == max_score]
        xy, score = tmp[0] if len(tmp) == 1 else random.choice(tmp)
        return score, xy
    else:
        return 0, None


def get_train_test_data(wave_data):
    train_data = [x for x in wave_data if x["date"] in TRAIN_DAYS]
    test_data = [x for x in wave_data if x["date"] not in TRAIN_DAYS]
    train_data.sort(key=lambda x: x["cid"])
    test_data.sort(key=lambda x: x["cid"])
    train_data = {k: list(v) for k, v in groupby(train_data, key=lambda x: x["cid"])}
    test_data = {k: list(v) for k, v in groupby(test_data, key=lambda x: x["cid"])}
    return train_data, test_data


def refine_match_with_physics(wave, params):
    stays, orders = wave["stays"], wave["orders"]
    oid2odr = {o["id"]: o for o in orders}
    is_ele = len(params) == 6

    def pre_t_stay(oids):
        unit2floors = defaultdict(list)
        for oid in oids:
            o = oid2odr[oid]
            unit2floors[o["unit"]].append(o["floor"])
        onum = len(oids)
        unit = len(unit2floors)
        wait1 = sum(max(floors) > 1 for floors in unit2floors.values())
        floor = sum(max(floors) - 1 for floors in unit2floors.values())
        wait2 = sum(len(set(floors) - {1}) for floors in unit2floors.values())
        if is_ele:
            return np.array([onum, unit - 1, wait1, wait2, floor, 1]) @ params
        else:
            return np.array([onum, unit - 1, floor, 1]) @ params

    def exchange_orders(giver, taker):
        t_g = giver["t"]
        loss_g = giver["t_pre"] - t_g
        t_t = taker["t"]
        loss_t = taker["t_pre"] - t_t
        last_loss = abs(loss_g) + abs(loss_t)
        oids_g = [x for x in sorted(list(zip(giver["oids_matched"], giver["match_scores"])), key=lambda x: x[1])]
        has_success = False
        for oid, score_g in oids_g:  
            o = oid2odr[oid]
            score, xy = match_s_o(taker, o)
            if score < max(MIN_MATCH_SCORE * 1.5, score_g * 0.1):
                continue
            t_pre_g = pre_t_stay([x for x in giver["oids_matched"] if x != oid])
            t_pre_t = pre_t_stay(taker["oids_matched"] + [oid])
            loss_taker = abs(t_pre_t - t_t)
            if not taker["oids_matched"]:
                loss_taker *= 0.3
            loss_giver = abs(t_pre_g - t_g)
            if len(giver["oids_matched"]) == 1:
                loss_giver = min(t_g, t_pre_g) * 2
            if loss_taker + loss_giver < last_loss:
                has_success = True
                for i, x in enumerate(giver["oids_matched"]):
                    if x == oid:
                        break
                for k in ["oids_matched", "match_xys", "match_scores"]:
                    giver[k] = giver[k][:i] + giver[k][i+1:]
                giver["t_pre"] = t_pre_g
                giver["is_taker"] = t_pre_g < t_g if giver["oids_matched"] else True
                taker["oids_matched"].append(oid)
                taker["match_xys"].append(xy)
                taker["match_scores"].append(score)
                taker["t_pre"] = t_pre_t
                taker["is_taker"] = t_pre_t < t_t
                last_loss = abs(t_pre_g - t_g) + abs(t_pre_t - t_t)
                if giver["is_taker"] != False and taker["is_taker"] != True:
                    return has_success
        return has_success

    stays_inlier = []
    for s in stays:
        s["t"] = t = s["trange"][1] - s["trange"][0]
        if len(s["oids_matched"]) == 0:
            s["is_taker"] = True
            s["t_pre"] = t
            stays_inlier.append(s)
        else:
            s["t_pre"] = t_pre = pre_t_stay(s["oids_matched"])
            if t - t_pre < 150:
                s["is_taker"] = t_pre < t
                stays_inlier.append(s)

    if config["miter"]:
        success_cnt = 0
        s12s = zip(stays_inlier[::-1][1:], stays_inlier[::-1])
        for s1, s2 in s12s:
            if s2["trange"][0] - s1["trange"][1] > 300 or s1["is_taker"] == s2["is_taker"]:
                continue
            if not s1["is_taker"] and s2["is_taker"]:
                success_cnt += exchange_orders(s1, s2)
            elif s1["is_taker"] and not s2["is_taker"]:
                success_cnt += exchange_orders(s2, s1)
   
    if not config["siter"]:
        return wave

    to_remove_j = []
    for i, s in enumerate(stays):
        t = s["t"]
        t_pre = s["t_pre"]
        ts, te = s["trange"]
        if not (s["oids_matched"] and t - t_pre < 150):
            continue
        td = min(10, (t_pre - t) * 0.1)
        t1 = ts - td
        t2 = te + td
        if td <= 0: 
            s["trange"] = [t1, t2]
            s["t"] = t2 - t1
        else:
            tmin = -1e6
            for j in range(i - 1, -1, -1):
                ss = stays[j]
                if ss["oids_matched"]:
                    tmin = ss["trange"][1]
                    break
            tmax = 1e6
            for j in range(i + 1, len(stays)):
                ss = stays[j]
                if ss["oids_matched"]:
                    tmax = ss["trange"][0]
                    break 
            if t1 < tmin and t2 < tmax:
                t2 += tmin - t1
                t2 = min(t2, tmax)
                t1 = tmin
            elif t2 > tmax and t1 > tmin:
                t1 -= t2 - tmax
                t1 = max(tmin, t1)
                t2 = tmax
            for j in range(i - 1, -1, -1):
                ss = stays[j]
                a, b = ss["trange"]
                if t1 >= b:
                    break
                else:
                    if ss["oids_matched"]:
                        t1 = b
                        break
                    else:
                        if t1 > a:
                            ss["trange"] = [a, t1]
                            ss["t"] = t1 - a
                            break
                        else:
                            to_remove_j.append(j)
            for j in range(i + 1, len(stays)):
                ss = stays[j]
                a, b = ss["trange"]
                if t2 <= a:
                    break
                else:
                    if ss["oids_matched"]:
                        t2 = a
                        break
                    else:
                        if t2 < b:
                            ss["trange"] = [t2, b]
                            ss["t"] = b - t2
                            break
                        else:
                            to_remove_j.append(j)
            s["trange"] = [t1, t2]
            s["t"] = t2 - t1
    wave["stays"] = [s for i, s in enumerate(stays) if i not in to_remove_j]

    return wave


def train_physics(waves, region):
    is_ele = region["is_elevator"]
    oid2odr = {o["id"]: o for w in waves for o in w["orders"]}

    stays = []
    for w in waves:
        poly = region["poly"]
        for s in w["stays"]:
            if len(s["oids_matched"]) > 0 and s["trange"][1] - s["trange"][0] < 900:
                p = Point(s["point"][:2])
                if p.within(poly):
                    stays.append(s)

    samples = []
    filter_cnt = 0
    for s in stays:
        unit2floors = defaultdict(list)
        oids = s["oids_matched"]
        for oid in oids:
            o = oid2odr[oid]
            unit2floors[o["unit"]].append(o["floor"])
        onum = len(oids)
        unit = len(unit2floors)
        wait1 = sum(max(floors) > 1 for floors in unit2floors.values())
        floor = sum(max(floors) - 1 for floors in unit2floors.values())
        wait2 = sum(len(set(floors) - {1}) for floors in unit2floors.values())
        tmin = TMIN_BASE + TMIN_ORDER * onum + TMIN_CUNIT * (unit - 1)
        tmax = TMAX_BASE + TMAX_ORDER * onum + TMAX_CUNIT * (unit - 1)
        if is_ele:
            tmin += TMIN_ELE_WAIT1 * wait1 + TMIN_ELE_WAIT2 * wait2 + TMIN_ELE * floor
            tmax += TMAX_ELE_WAIT1 * wait1 + TMAX_ELE_WAIT2 * wait2 + TMAX_ELE * floor  
        else:
            tmin += TMIN_STAIR * floor
            tmax += TMAX_STAIR * floor
        t_stay = s["trange"][1] - s["trange"][0]
        if not tmin < t_stay < tmax:
            filter_cnt += 1
            continue
        if is_ele:
            samples.append([onum, unit - 1, wait1, wait2, floor, t_stay])
        else:
            samples.append([onum, unit - 1, floor, t_stay])

    samples = np.array(samples)
    X, Y = samples[:, :-1], samples[:, -1]
    regs = [
        linear_model.LinearRegression(positive=True), 
        linear_model.Ridge(alpha=0.1, positive=True)
    ]
    results = []
    for reg in regs:
        mask = np.ones_like(Y, dtype=bool)
        cnt = 0
        while(True):
            X_train, Y_train = X[mask], Y[mask]
            reg.fit(X_train, Y_train)
            params = list(reg.coef_) + [reg.intercept_]

            losses = reg.predict(X) - Y
            last_mask = mask
            mask = np.abs(losses) < 150

            inlier_losses = reg.predict(X_train) - Y_train
            print("inlier loss:", 
                  np.mean([abs(x) for x in inlier_losses]), 
                  np.mean([abs(x) / y for x, y in zip(inlier_losses, Y_train)]), 
                  np.mean(Y_train), 
                  np.mean([abs(x) for x in inlier_losses]) / np.mean(Y_train))

            if np.all(mask == last_mask):
                break
            cnt += 1
            if cnt > 10:
                break
        
        results.append((params, np.mean([abs(x) for x in inlier_losses])))

    params = min(results, key=lambda x: x[-1])[0]
    return params


def evaluate(wave, actions, do_plot=False, params=None):
    global dfr
    stays = wave["stays"]
    stay_ranges = [[s["trange"], len(s["oids_matched"]) > 0] for s in stays]
    st, et = wave["wave"]
    oid2odr = {o["id"]: o for o in wave["orders"]}

    actions = deepcopy(actions)
    for i, a in enumerate(actions):
        if a["et"] > st:
            a["st"] = max(a["st"], st)
            break
    for j in range(len(actions)-1, -1, -1):
        a = actions[j]
        if a["st"] < et:
            a["et"] = min(a["et"], et)
            break
    actions = actions[i:j+1]

    def get_coarse_time(st, et):
        nonlocal stay_ranges
        work = 0
        rest = 0
        for (t1, t2), is_work in stay_ranges:
            if t2 <= st:
                continue
            if t1 >= et:
                break
            t = min(et, t2) - max(st, t1)
            assert t >= 0
            if is_work:
                work += t
            else:
                rest += t
        walk = et - st - work - rest
        assert walk >= 0
        return work, rest, walk

    total_t = 0
    correct_t = 0
    dtt, dct, rtt, rct, wtt, wct = 0, 0, 0, 0, 0, 0
    for a in actions:
        st, et = a["st"], a["et"]
        t = a["et"] - a["st"]
        work, rest, walk = get_coarse_time(st, et)
        if a["act"] == WORK:
            total_t += t
            dtt += t
            correct_t += work
            dct += work
        elif a["act"] == OTHER:
            total_t += t
            wtt += t
            correct_t += rest + walk
            wct += rest + walk
        elif a["act"] == REST:
            total_t += t
            rtt += t
            correct_t += rest
            rct += rest

    if not params:
        return {"acc": np.array([correct_t, total_t, dct, dtt, wct, wtt, rct, rtt])}

    oid2tf_gt = {}
    for a in actions:
        if a["oids"]:
            _dict, t_stones = fine_rec_abm(a["st"], a["et"], [oid2odr[oid] for oid in a["oids"]], params)
            oid2tf_gt.update(_dict)
            a["t_stones"] = t_stones
    
    oid2tf = {}
    stay_fine_ranges = []
    if config["fine"] == "deep":
        date = wave["date"]
        t = date % 7
        if t == 0:
            t = 7
        is_weekend = t > 5
        slot = int((np.mean(wave["wave_traj"]) - 8 * 3600) / 3 / 3600)
        slot = max(0, min(4, slot))
    for s in stays:
        if s["oids_matched"]:
            odrs = [oid2odr[oid] for oid in s["oids_matched"]]
            if config["fine"] == "mid":
                _oid2tf, _ranges = fine_rec_mid(*s["trange"], odrs)
            elif config["fine"] == "unf":
                _oid2tf, _ranges = fine_rec_unf(*s["trange"], odrs)
            elif config["fine"] == "smt":
                _oid2tf, _ranges = fine_rec_smt(*s["trange"], odrs)
            elif config["fine"] == "deep":
                _oid2tf, _ranges = dfr.infer(*s["trange"], odrs, wave["cid"], is_weekend, slot)
            elif config["fine"] == "abm":
                _oid2tf, t_stones = fine_rec_abm(*s["trange"], odrs, params)
                last_et = s["trange"][0]
                _ranges = []
                for t, atp in t_stones:
                    _ranges.append([[last_et, t], atp])
                    last_et = t
            assert abs(s["trange"][0] - _ranges[0][0][0]) < 0.1
            assert abs(s["trange"][1] - _ranges[-1][0][1]) < 0.1
            for (st, et), _ in _ranges:
                assert et - st >= 0
            oid2tf.update(_oid2tf)
            stay_fine_ranges += _ranges
        else:
            stay_fine_ranges.append([s["trange"], NOT_WORK])

    def get_fine_time(st, et):
        nonlocal stay_fine_ranges
        up, down, unit, deliver, arrange = 0, 0, 0, 0, 0
        for (t1, t2), atp in stay_fine_ranges:
            if t2 <= st:
                continue
            if t1 >= et:
                break
            t = min(et, t2) - max(st, t1)
            assert t >= 0
            if atp == UP:
                up += t
            elif atp == DOWN:
                down += t
            elif atp == UNIT:
                unit += t
            elif atp == DELIVER:
                deliver += t
            elif atp == ARRANGE:
                arrange += t
            else:
                assert atp == NOT_WORK
        return up, down, unit, deliver, arrange

    total_fine_t = total_t - dtt
    correct_fine_t = correct_t - dct
    uptt, upct, dott, doct, utt, uct, fdtt, fdct, att, act = [0 for _ in range(10)]
    for a in actions:
        if a["oids"]:
            st, et = a["st"], a["et"]
            total_fine_t += a["et"] - a["st"]
            t_stones = a["t_stones"]
            last_et = st
            for t, atp in t_stones:
                up, down, unit, deliver, arrange = get_fine_time(last_et, t)
                tt = t - last_et
                last_et = t
                if atp == UP:
                    uptt += tt
                    correct_fine_t += up
                    upct += up
                elif atp == DOWN:
                    dott += tt
                    correct_fine_t += down
                    doct += down
                elif atp == UNIT:
                    utt += tt
                    correct_fine_t += unit
                    uct += unit
                elif atp == DELIVER:
                    fdtt += tt
                    correct_fine_t += deliver
                    fdct += deliver
                elif atp == ARRANGE:
                    att += tt
                    correct_fine_t += arrange
                    act += arrange
                else:
                    assert False
    assert abs(dtt - (uptt + dott + utt + fdtt + att)) < 0.1

    losses = []
    for oid, t_gt in oid2tf_gt.items():
        losses.append(oid2tf[oid] - t_gt)

    pure_cft = correct_fine_t - (correct_t - dct)
    pure_ctt = total_fine_t - (total_t - dtt)
    assert abs(upct + doct + uct + fdct + act - pure_cft) < 0.1
    assert abs(uptt + dott + utt + fdtt + att - pure_ctt) < 0.1
    return {
        "acc": np.array([correct_t, total_t, dct, dtt, wct, wtt, rct, rtt]),
        "dtf": losses,
        "acc_fine": np.array([
            correct_fine_t, total_fine_t, 
            upct, uptt, doct, dott, uct, utt, fdct, fdtt, act, att,
            pure_cft, pure_ctt
            ])
    }


def main(train_waves, test_waves, region, actions):
    waves = [b for a in [train_waves, test_waves] for b in a]
    metrics_out = []
    for n_iter_out in tqdm(range(N_ITER_OUT)):
        if n_iter_out == 0 or config["piter"]:
            params = train_physics(train_waves, region)
        metrics_in = []
        for n_iter_in in range(N_ITER_IN):
            if n_iter_out == 0 and n_iter_in == 0:
                metrics = [evaluate(w, actions, params=params) for w in test_waves]
                metrics_in.append({
                    "acc": sum(m["acc"] for m in metrics),
                    "mr": sum(m["mr"] for m in metrics),
                    "dtf": sum([m["dtf"] for m in metrics], []),
                    "acc_fine": sum(m["acc_fine"] for m in metrics),
                })
            for w in waves:
                refine_match_with_physics(w, params)
            metrics = [evaluate(w, actions, params=params) for w in test_waves]
            metrics_in.append({
                "acc": sum(m["acc"] for m in metrics),
                "mr": sum(m["mr"] for m in metrics),
                "dtf": sum([m["dtf"] for m in metrics], []),
                "acc_fine": sum(m["acc_fine"] for m in metrics),
            })
        metrics_out.append(metrics_in)

    return metrics_out


if __name__ == "__main__":
    cid2label_actions = pickle.load(open("dataset/gt_label_actions.pkl", "rb"))
    cid2region = pickle.load(open("dataset/cid2region.pkl", "rb"))
    config_all = yaml.load(open("config.yml", "r"), Loader=yaml.SafeLoader)
    cids = sorted(list(cid2label_actions.keys()))
    cid2idx = {cid: i for i, cid in enumerate(cids)}
    dfr = DeepFineRec(cid2idx)
    dfr.load_model("log/DeepFineRec_230609_095452/278.pt")
    
    modes = ["Ours", "MS+mid", "MS+unf", "MS+smt", "MS+deep", 
        "MT+mid", "MT+unf", "MT+smt", "MT+deep", 
        "DTInf+mid", "DTInf+unf", "DTInf+smt", "DTInf+deep", 
        "MetaSTP+mid", "MetaSTP+unf", "MetaSTP+smt", "MetaSTP+deep"]
    for MODE in modes:
        print(MODE)
        config = config_all[MODE]

        if config["stay_ref"]:
            wave_data = pickle.load(open("dataset/wave_data_corrected.pkl", "rb"))
        else:
            wave_data = pickle.load(open("dataset/wave_data_no_stay_ref_corrected.pkl", "rb"))

        wave_data = coarse_rec(wave_data, config["adr_ref"], config["coarse"], config["coarse_post"])

        train_data, test_data = get_train_test_data(wave_data)

        for cid in cids:
            actions = cid2label_actions[cid]
            region = cid2region[cid]
            train_waves = train_data[cid]
            test_waves = test_data[cid]
            waves = [b for a in [train_waves, test_waves] for b in a]
            params = train_physics(waves, region)
            for w in test_waves:
                evaluate(w, actions, params=params)

        cid2metrics = {}
        for cid in cids:
            actions = cid2label_actions[cid]
            region = cid2region[cid]
            metrics = main(train_data[cid], test_data[cid], region, actions)
            cid2metrics[cid] = [y for x in metrics for y in x]
