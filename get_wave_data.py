import pickle
import random
from collections import Counter
from math import ceil
from math import pi as PI

import numpy as np
from shapely.geometry import Point
from tqdm import tqdm

from constants import *

NUM_TOLERANCE = 5
START_END_IGNORE = 6
START_END_SHIFT_GAP = 1800 
WAVE_GAP = 3600
WAVE_PORTION = 0.07
WAVE_EXTEND = 900
TRAJ_GAP = 120
TRAJ_COVER_GATE = 0.7
FILTER_WAVE_LENGTH = 1800
FILTER_WAVE_ORDER_NUM = 20 
FILTER_WAVE_NOISE = 0.15
NEAR_STA_DIS2 = 120 ** 2

DIS_REGION = 100
V_NOISE = 12

TRAJ_UPSAMPLE = 5
D_STAY = 30 
T_STAY = 60
D_STAY_MERGE = 45
T_STAY_MERGE = 60
FLAT_GATE = 0.0225
POINT_AREA = 0.1
POINT_LENGTH = 2 * PI * (POINT_AREA / PI) ** 0.5
GRID_LEN = 300
SLOT_LEN = 3600
random.seed(233)


def cum(ts, for_plot=False):
    """
    不知道是啥，貌似是生成时间轴数据
    """
    num0 = len([t for t in ts if t <= 0])
    t_nums = sorted(list(Counter([t for t in ts if t > 0]).items()), key=lambda x:x[0])
    if for_plot: 
        points = [(0, num0)]
    else:
        points = []
    cnt = num0
    for t, n in t_nums:
        if for_plot:
            points.append((t, cnt))
        cnt += n
        points.append((t, cnt))
    return points


def find_cum_at_t(points, t_target):
    """
    不知道是啥
    """
    if t_target < points[0][0]:
        return 0
    for i, (t, n) in enumerate(points):
        if t == t_target:
            return n
        if t > t_target:
            return points[i-1][1]
    return points[-1][1]


def get_waves(orders):
    """
    貌似是对订单进行波次划分
    """
    start_times = [o["start_time"] for o in orders if o["type"] == ORDER_DELIVER]
    finish_times = [o["finish_time"] for o in orders]
    receive_points = cum(start_times)          #时间轴数据
    finish_points = cum(finish_times)
 
    ends = []
    jump_idx = 0
    last_n1 = 0
    last_end = None
    for i, (t1, n1) in enumerate(finish_points):  
        if i < jump_idx:
            continue
        if i == len(finish_points) - 1:  
            break
        for j in range(i+1, len(finish_points)):
            t2, n2 = finish_points[j]
            if n2 - n1 >= NUM_TOLERANCE:
                break
        else:
            if t2 - t1 > WAVE_GAP and n1 - last_n1 > len(orders) * (WAVE_PORTION + 0.03):
                last_end = t1
            break
        
        if t2 - t1 > WAVE_GAP and n1 - last_n1 > len(orders) * WAVE_PORTION and \
          find_cum_at_t(receive_points, t2) - find_cum_at_t(receive_points, t1) > len(orders) * WAVE_PORTION:
            ends.append(t1)
            last_n1 = n1
            jump_idx = j + 1
        else:
            jump_idx = i + 1
    if ends:
        n_last_finish = find_cum_at_t(finish_points, ends[-1])
        if finish_points[-1][1] - n_last_finish > START_END_IGNORE:
            if last_end is not None:
                ends.append(last_end)
            else:
                ends.append(finish_points[-1][0])
    
    starts = []
    jump_idx = len(finish_points)-1
    next_n1 = len(orders)
    first_start = None
    for i in range(len(finish_points)-1, -1, -1):  
        t1, n1 = finish_points[i]
        if i > jump_idx:
            continue
        if i == 0:
            break
        for j in range(i-1, -1, -1):
            t2, n2 = finish_points[j]
            if n1 - n2 >= NUM_TOLERANCE:
                break
        else:
            if t2 - t1 > WAVE_GAP and next_n1 - n1 > len(orders) * (WAVE_PORTION + 0.03):
                first_start = t1
            break
        if t1 - t2 > WAVE_GAP and next_n1 - n1 > len(orders) * WAVE_PORTION and \
          find_cum_at_t(receive_points, t1) - find_cum_at_t(receive_points, t2) > len(orders) * WAVE_PORTION:
            starts.append(t1)
            next_n1 = n1
            jump_idx = j - 1
        else:
            jump_idx = i - 1
    starts = starts[::-1]
    if starts:
        n_first_start = find_cum_at_t(finish_points, starts[0] - 1)
        if n_first_start > START_END_IGNORE:
            if first_start is not None:
                starts = [first_start] + starts
            else:
                starts = [finish_points[0][0]] + starts
    
    if len(starts) == 0 and len(ends) == 0 and len(orders) > 30:
        starts = [first_start if first_start is not None else finish_points[0][0]]
        ends = [last_end if last_end is not None else finish_points[-1][0]]

    try:
        assert len(starts) == len(ends)
        waves = list(zip(starts, ends))
        last_e = -999
        for s, e in waves:
            assert e > s
            assert s > last_e
            last_e = e
    except:
        if len(starts) < len(ends):
            starts += [0] * (len(ends) - len(starts))
        else:
            ends += [0] * (len(starts) - len(ends))
        return list(zip(starts, ends)), None
    
    return waves


def get_traj_tm_cover(traj):
    """
    貌似是将订单数据划分成cover份
    """
    ts = [p[-1] for p in traj]
    cover = 0
    for t1, t2 in zip(ts, ts[1:]):
        d = t2 - t1
        if d < TRAJ_GAP:
            cover += d
    return cover


def cal_v(p1, p2):
    """
    计算速度
    """
    (x1, y1, t1), (x2, y2, t2) = p1, p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / (t2 - t1)


def cal_dis(p1, p2):
    """
    计算距离
    """
    (x1, y1), (x2, y2) = p1, p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def denoise(traj):
    """
    对轨迹去噪。
    有几种规则：
    """
    to_remove = []
    #1.轨迹点是否在区域内
    for i, p in enumerate(traj):
        p = Point(p[:2])
        for r in regions.values():             #regions = pickle.load(open("dataset/regions.pkl", "rb"))
            if p.distance(r["poly"]) < DIS_REGION:
                break
        else:
            to_remove.append(i)
    traj = [p for i, p in enumerate(traj) if i not in set(to_remove)]

    #2.使用doing in one go的方法，相邻轨迹点的速度不能太大
    noise_idxs = []
    traj_filter = []
    for i, p1 in enumerate(traj):
        for j in range(i + 1, i + 3):
            p2 = traj[j]
            if cal_v(p1, p2) >= V_NOISE:
                noise_idxs.append(i)
                break
        else:
            traj_filter.append(p1)
            break
    jump_idx = i + 1

    for i, p in enumerate(traj):
        if i < jump_idx:
            continue
        if cal_v(traj_filter[-1], p) < V_NOISE:
            traj_filter.append(p)
        else:
            noise_idxs.append(i)

    print("noise:", len(noise_idxs), "/", len(traj))
    return traj_filter, len(noise_idxs) / len(traj)


def get_stay_points(traj, do_ref=True):
    """
    寻找驻留点
    """
    def upscale_traj(traj, sample_gap=5):
        """
        上采样轨迹点
        """
        traj_new = [traj[0]]
        last_x, last_y, last_t = traj[0]
        for x, y, t in traj[1:]:
            if t - last_t > sample_gap:
                section_num = ceil((t - last_t) / sample_gap)
                delta_x, delta_y, delta_t = x - last_x, y - last_y, t - last_t
                for i in range(1, section_num):
                    p = i / section_num 
                    traj_new.append((last_x + p*delta_x, last_y + p*delta_y, last_t + p*delta_t))
            traj_new.append((x, y, t))
            last_x, last_y, last_t = x, y, t
        return traj_new
    
    traj = upscale_traj(traj, sample_gap=TRAJ_UPSAMPLE)
    stays = []
    jump_idx = 0
    for i, (x, y, t) in enumerate(traj):
        if i < jump_idx:
            continue
        for j in range(i+1, len(traj)):
            if traj[j][-1] - t >= T_STAY:
                break
        else:
            break
        for k in range(j, i, -1):
            x1, y1 = traj[k][:2]
            if (x - x1) ** 2 + (y - y1) ** 2 > D_STAY ** 2:
                break
        else:
            if j == len(traj) - 1:
                k = j + 1
            else:
                for k in range(j+1, len(traj)):
                    x1, y1 = traj[k][:2]
                    if (x - x1) ** 2 + (y - y1) ** 2 > D_STAY ** 2:
                        break
                else:
                    k += 1
            if not do_ref:
                stays.append([i, k-1])
                jump_idx = k
            else:
                ps = [p[:2] for p in traj[i:k]]
                xs, ys = zip(*ps)
                center = (np.mean(xs), np.mean(ys))
                r = np.mean([cal_dis(p, center) for p in ps])
                r = max(D_STAY / 2, r)
                assert r <= D_STAY
                for j in range(i, k):  
                    if cal_dis(traj[j][:2], center) <= r:
                        break
                else:
                    assert False
                i_new = j
                if k == len(traj):
                    k_new = k
                else:
                    for j in range(k, len(traj)):
                        if cal_dis(traj[j][:2], center) > r:
                            break
                    else:
                        j += 1
                    k_new = j
                assert i_new < k_new
                ps = [p[:2] for p in traj[i_new:k_new]]
                xs, ys = zip(*ps)
                center = (np.mean(xs), np.mean(ys))
                r = np.mean([cal_dis(p, center) for p in ps])
                r = max(D_STAY / 2, r)
                assert r <= D_STAY
                if k_new - i_new > 1:
                    for j in range(k_new - 1, i_new, -1):  
                        if cal_dis(traj[j][:2], center) <= r:
                            break
                    else:
                        assert False
                    k_new = j + 1
                assert i_new < k_new

                ps = [p[:2] for p in traj[i_new:k_new]]
                xs, ys = zip(*ps)
                center = (np.mean(xs), np.mean(ys))
                r = np.mean([cal_dis(p, center) for p in ps])
                r = max(D_STAY * 2 / 3, r)
                assert r <= D_STAY
                for j in range(i_new, k_new):  
                    if cal_dis(traj[j][:2], center) <= r:
                        break
                else:
                    assert False
                i_new = j
                if k_new - i_new > 1:
                    for j in range(k_new - 1, i_new, -1):  
                        if cal_dis(traj[j][:2], center) <= r:
                            break
                    else:
                        assert False
                    k_new = j + 1
                
                if traj[k_new - 1][-1] - traj[i_new][-1] > T_STAY * 2 / 3:
                    stays.append([i_new, k_new-1])
                    jump_idx = k_new
            
    to_remove = []
    for i, (s, e) in enumerate(stays):
        ps = traj[s:e+1]
        xs, ys, _ = zip(*ps)
        x, y = np.mean(xs), np.mean(ys)
        if (x - X_STA) ** 2 + (y - Y_STA) ** 2 < NEAR_STA_DIS2:
            to_remove.append(i)
    stays = [x for i, x in enumerate(stays) if i not in to_remove]
    orig_stay_num = len(stays)

    if do_ref and len(stays) > 1:
        stays_merge = [stays[0]]
        x, y = traj[stays[0][0]][:2]
        last_e = stays[0][1]
        last_t = traj[last_e][-1]
        for s, e in stays[1:]:
            if traj[s][-1] - last_t < T_STAY_MERGE:
                for k in range(last_e + 1, e + 1):
                    x1, y1 = traj[k][:2]
                    if (x - x1) ** 2 + (y - y1) ** 2 > D_STAY_MERGE ** 2:
                        stays_merge.append([s, e])
                        x, y = traj[s][:2]
                        last_e = e
                        last_t = traj[e][-1]
                        break
                else:
                    stays_merge[-1][1] = e
                    last_e = e
                    last_t = traj[e][-1]
            else:
                stays_merge.append([s, e])
                x, y = traj[s][:2]
                last_e = e
                last_t = traj[e][-1]
        stays = stays_merge
    merged_stay_num = len(stays)

    last_e = -1
    for s, e in stays:
        assert e > s
        assert s > last_e, (s, last_e)
        last_e = e

    stays = [{"se": se} for se in stays]
    for i, x in enumerate(stays):
        s, e = x["se"]
        ps = traj[s:e+1]
        x["traj"] = ps
        xs, ys, ts = zip(*ps)
        x_avg, y_avg, t_avg = np.mean(xs), np.mean(ys), np.mean(ts)
        x["point"] = (round(x_avg, 6), round(y_avg, 6), t_avg)
        x["trange"] = (ts[0], ts[-1])
    stay_idxs = set(sum([list(range(x["se"][0], x["se"][1]+1)) for x in stays], []))
    print("stay points:", orig_stay_num, merged_stay_num, len(stay_idxs), "/", len(traj))
        
    return traj, stays


def read_wave_data(orig_data, do_ref=True, do_plot=False):
    """
    貌似是划分波次的
    """
    wave_data = []
    fail_waves = 0
    short_wave = 0
    long_wave = 0
    bad_traj_tm = 0
    bad_traj_noise = 0
    good_traj = 0
    for data in tqdm(orig_data):
        waves = get_waves(data["orders"])
        if not waves or waves[-1] is None:
            print("fail wave:", data["cid"], data["date"])
            fail_waves += 1
            continue
        waves_long = [[s, e] for s, e in waves if e - s > FILTER_WAVE_LENGTH]
        traj = data["traj"]
        for i, (s, e) in enumerate(waves_long):
            s_new = s
            for x, y, t in traj:
                if s < t < s + 1800:
                    if (x - X_STA) ** 2 + (y - Y_STA) ** 2 < NEAR_STA_DIS2:
                        s_new = t
                    else:
                        break
            e_new = e
            for x, y, t in traj[::-1]:
                if e - 1800 < t < e:
                    if (x - X_STA) ** 2 + (y - Y_STA) ** 2 < NEAR_STA_DIS2:
                        e_new = t
                    else:
                        break
            waves_long[i] = [s_new, e_new]
        waves_long = [[s, e] for s, e in waves_long if e - s > FILTER_WAVE_LENGTH]
        short_wave += len(waves) - len(waves_long)
        long_wave += len(waves_long)

        waves_traj = []
        waves_order = []
        for i, (s, e) in enumerate(waves_long):
            if i == 0:
                if i + 1 < len(waves_long):
                    mid = (waves_long[i+1][0] + e) / 2
                    waves_traj.append([max(s - WAVE_EXTEND, 0), min(e + WAVE_EXTEND, mid, 86399)])
                    waves_order.append([0, mid])
                    last_mid = mid
                else:
                    waves_traj.append([max(s - WAVE_EXTEND, 0), min(e + WAVE_EXTEND, 86399)])
                    waves_order.append([0, 86399])
            else:
                if i + 1 < len(waves_long):
                    mid = (waves_long[i+1][0] + e) / 2
                    waves_traj.append([max(s - WAVE_EXTEND, last_mid, 0), min(e + WAVE_EXTEND, mid, 86399)])
                    waves_order.append([last_mid, mid])
                    last_mid = mid
                else:
                    waves_traj.append([max(s - WAVE_EXTEND, last_mid, 0), min(e + WAVE_EXTEND, 86399)])
                    waves_order.append([last_mid, 86399])
        
        last_e = -999
        for s, e in waves_traj:
            assert e > s
            assert s >= last_e
            last_e = e
        last_e = -999
        for s, e in waves_order:
            assert e > s
            assert s >= last_e
            last_e = e
        assert len(waves_long) == len(waves_traj) == len(waves_order)

        cid = data["cid"]
        date = data["date"]
        orders = data["orders"]
        traj = data["traj"]

        wave_idx = 0
        for (s, e), (st, et), (so, eo) in zip(waves_long, waves_traj, waves_order):
            tj = [p for p in traj if st < p[-1] <= et]
            if get_traj_tm_cover(tj) / (e - s) < TRAJ_COVER_GATE:
                bad_traj_tm += 1
                continue
            
            tj_denoise, noise_portion = denoise(tj)
            if noise_portion > FILTER_WAVE_NOISE:
                bad_traj_noise += 1
                continue
            
            good_traj += 1
            tj_new, stays = get_stay_points(tj_denoise, do_ref)
            odrs = [o for o in orders if so < o["finish_time"] <= eo]
            odrs.sort(key=lambda x: x["finish_time"])
            wave_data.append({
                "cid": cid,
                "date": date,
                "orders": odrs,  
                "traj": tj_new,
                "stays": stays,
                "wave": (s, e),
                "wave_traj": (st, et),
                "wave_order": (so, eo),
                "wave_idx": wave_idx,
                "is_morning": st < 12 * 3600
            })
            wave_idx += 1
    return wave_data


if __name__ == "__main__":
    orig_data = pickle.load(open("dataset/orig_data.pkl", "rb"))
    wave_data = read_wave_data(orig_data, do_ref=True)
    pickle.dump(wave_data, open("dataset/wave_data.pkl", "wb"))

    orig_data = pickle.load(open("dataset/orig_data.pkl", "rb"))
    wave_data = read_wave_data(orig_data, do_ref=False)
    pickle.dump(wave_data, open("dataset/wave_data_no_stay_ref.pkl", "wb"))
