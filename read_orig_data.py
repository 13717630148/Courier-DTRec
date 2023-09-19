import json
import pickle
import time
from collections import defaultdict

import numpy as np
from coord_convert.transform import gcj2wgs
from shapely.geometry import Point
from tqdm import tqdm

from constants import *

FILTER_ORDER_NUM = 50
DIS_REGION = 100
MATCH_TO_NEAR_BD = 20


def get_nearest_bid(p):
    dis_id = []
    for bid, b in buildings.items():
        dis = p.distance(b["poly"])
        if dis == 0:
            return bid, 0
        dis_id.append([dis, bid])
    else:
        dis, bid = min(dis_id, key=lambda x: x[0])
        return bid, dis
    

def read(MONTH, po, pt, dates, date_offset, uid_offset):
    adsid2gps = json.load(open("dataset/adsid2gaode_gps.json"))
    adsid2xy = {
        int(adsid): 
        projector(*gcj2wgs(*gps)) 
        for adsid, gps in adsid2gps.items() if gps
    }

    data = []
    tp_map = {
        "派送": ORDER_DELIVER, 
        "C揽": ORDER_CPICK, 
        "B揽": ORDER_BPICK}
    uid = 0
    uid += uid_offset
    throw_out_of_region = 0
    match_xy_to_bd = 0
    no_bd = 0
    large_dis_xy_bd = 0

    bc_cnt = 0
    bc_throw_cnt = 0
    for date in tqdm(dates):
        t_base = time.mktime(time.strptime(f"2022-{MONTH:02}-{date:02} 00:00:00", "%Y-%m-%d %H:%M:%S"))
        cid2orders = defaultdict(list)
        lines = open(po + f"{date:02}.csv").readlines()[1:]
        for l in lines:
            l = l.strip("\n").split(",")
            if MONTH == 8:
                l = l[0:2] + l[3:] 
            cid = int(l[2])
            tp = tp_map[l[4]]
            if l[4] != "派送":
                bc_cnt += 1

            xy = projector(*gcj2wgs(float(l[6]), float(l[5])))
            xy = (round(xy[0], 6), round(xy[1], 6))
            address_id = int(l[7])
            address_xy = adsid2xy.get(address_id, None)
            st, et, ddl = float(l[8]), float(l[9]), float(l[10])
            if not 0 <= et < 86400:
                continue
            floor = int(l[11]) if l[11] else -1
            try:
                unit = int(l[12])
            except:
                unit = -1
            bid = int(l[13])

            if address_xy:
                p = Point(address_xy)
                for r in regions.values():
                    if p.distance(r["poly"]) < DIS_REGION:
                        break
                else:
                    address_xy = None
            p = Point(xy)
            for r in regions.values():
                if p.distance(r["poly"]) < DIS_REGION:
                    break
            else:
                if address_xy:
                    throw_out_of_region += 1
                    if l[4] != "派送":
                        bc_throw_cnt += 1
                    continue
            if bid == -1:
                bid_new, dis = get_nearest_bid(p)
                if dis < MATCH_TO_NEAR_BD:
                    bid = bid_new
                    match_xy_to_bd += 1
                else:
                    no_bd += 1
            if bid != -1:
                dis_xy_bd = p.distance(buildings[bid]["poly"])
                if dis_xy_bd > MATCH_TO_NEAR_BD:
                    large_dis_xy_bd += 1
            else:
                dis_xy_bd = -1

            cid2orders[cid].append({
                "id": uid,
                "type": tp,
                "start_time": st,
                "finish_time": et,
                "ddl_time": ddl,
                "building_id": bid,
                "floor": floor,
                "unit": unit,
                "address_id": address_id,
                "address_xy": address_xy,
                "xy": xy,
                "dis_xy_bd": dis_xy_bd
            })
            uid += 1
        for cid, orders in cid2orders.items():
            orders.sort(key=lambda x: x["finish_time"])
            orders = [o for o in orders if o["start_time"] <= o["finish_time"]]
            cid2orders[cid] = orders

        cid2traj = defaultdict(list)
        lines = open(pt + f"{date:02}.csv").readlines()[1:]
        for l in lines:
            if MONTH == 11:
                l = l.strip("\n").split(",")
                x, y = projector(*gcj2wgs(float(l[2]), float(l[1])))
                p = Point((x, y))
                for r in regions.values():
                    if p.distance(r["poly"]) < DIS_REGION:
                        break
                else:
                    continue
                t = time.mktime(time.strptime(l[3], "%Y-%m-%d %H:%M:%S")) - t_base
                assert 0 <= t < 86400
                cid = int(l[4])
                assert l[-1] == f"2022-{MONTH:02}-{date:02}"
                cid2traj[cid].append((x, y, t))
            else:
                l = l.strip("\n").split(",")
                x, y = projector(*gcj2wgs(float(l[6]), float(l[5])))
                p = Point((x, y))
                for r in regions.values():
                    if p.distance(r["poly"]) < DIS_REGION:
                        break
                else:
                    continue
                t = time.mktime(time.strptime(l[4], "%Y-%m-%d %H:%M:%S")) - t_base
                assert 0 <= t < 86400
                cid = int(l[2])
                assert l[-1] == f"2022-{MONTH:02}-{date:02}"
                cid2traj[cid].append((x, y, t))
        for cid, traj in cid2traj.items():
            traj = list(set(traj))
            t2xys = defaultdict(list)
            for x, y, t in traj:
                t2xys[t].append((x, y))
            t_xys = list(t2xys.items())
            t_xys.sort(key=lambda x: x[0])
            traj = []
            multiple_cnt= 0
            for t, xys in t_xys:
                if len(xys) == 1:
                    traj.append((*xys[0], float(t)))
                else:
                    xs, ys = zip(*xys)
                    x, y = np.mean(xs), np.mean(ys)
                    dis = np.mean([((x1 - x) ** 2 + (y1 - y) ** 2) ** 0.5 for x1, y1 in xys])
                    if dis < 10:
                        traj.append((x, y, float(t)))
                    else:
                        multiple_cnt += 1
            cid2traj[cid] = traj
            print(cid, "multiple coords:", multiple_cnt, "/", multiple_cnt + len(traj))

        for cid, orders in cid2orders.items():
            if cid not in cid2traj:
                print(cid, "no traj")
                continue
            if date != 1 and len(orders) < FILTER_ORDER_NUM:
                print(cid, f"orders < {FILTER_ORDER_NUM}:", len(orders))
                continue
            data.append({
                "cid": int(cid),
                "date": date + date_offset,
                "orders": orders,
                "traj": cid2traj[cid]
            })

    return data


if __name__ == "__main__":
    MONTH = 9
    po = f"dataset/2022-{MONTH:02}-"
    pt = f"dataset/gps_2022-{MONTH:02}-"
    dates = list(range(1, 7))
    date_offset = 900
    uid_offset = 0
    data9 = read(MONTH, po, pt, dates, date_offset, uid_offset)

    MONTH = 8
    po = f"dataset/2022-{MONTH:02}-"
    pt = f"dataset/gps_2022-{MONTH:02}-"
    dates = list(range(1, 32))
    date_offset = 800
    uid_offset = int(1e5)
    data8 = read(MONTH, po, pt, dates, date_offset, uid_offset)

    pickle.dump(data9 + data8, open("dataset/orig_data.pkl", "wb"))
