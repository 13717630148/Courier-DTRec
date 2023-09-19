import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from setproctitle import setproctitle
from tqdm import tqdm

from constants import (ARRANGE, DELIVER, DOWN, NOT_WORK, ORDER_BPICK,
                       ORDER_CPICK, ORDER_DELIVER, UNIT, UP)


def fine_rec_abm(st, et, odrs, params):
    is_ele = len(params) == 6
    if is_ele:
        to, tu, tw1, tw2, te, tb = params
        te /= 2
    else:
        to, tu, tf, tb = params
        tfu, tfd = tf * 2 / 3, tf / 3
    unit2odrs = defaultdict(list)
    oid2t = {}
    t = tb / 2
    t_stones = [[t, ARRANGE]]
    for odr in odrs:
        unit2odrs[odr["unit"]].append(odr)
    for i, odrs in enumerate(unit2odrs.values()):
        floor2odrs = defaultdict(list)
        for odr in odrs:
            floor2odrs[odr["floor"]].append(odr)
        floor_odrs = sorted(list(floor2odrs.items()), key=lambda x:x[0])
        last_floor = 1
        already_wait = False
        for floor, odrs in floor_odrs:
            up_num = floor - last_floor
            if up_num > 0:
                if is_ele:
                    t_wait = tw2 if already_wait else tw1
                    use_time = t_wait + te * up_num
                    already_wait = True
                else:
                    use_time = tfu * up_num
                t += use_time
                last_floor = floor
                t_stones.append([t, UP])
            for odr in odrs:
                t += to
                oid2t[odr["id"]] = t
            t_stones.append([t, DELIVER])
        down_num = last_floor - 1
        if down_num > 0:
            if is_ele:
                use_time = tw2 + te * down_num
            else:
                use_time = tfd * down_num
            t += use_time
            last_floor = 1
            t_stones.append([t, DOWN])
        if i < len(unit2odrs) - 1:
            t += tu
            t_stones.append([t, UNIT])
    t += tb / 2
    t_stones.append([t, ARRANGE])
    
    p = (et - st) / t  
    oid2t = {oid: st + x * p for oid, x in oid2t.items()}
    for x in t_stones:
        x[0] = st + x[0] * p

    return oid2t, t_stones


def fine_rec_mid(st, et, odrs):
    t_mid = (st + et) / 2
    return {o["id"]: t_mid for o in odrs}, [[[st, et], NOT_WORK]]


def fine_rec_unf(st, et, odrs):
    t = (et - st) / (len(odrs) + 1)
    ts = [st + t * i for i in range(1, len(odrs) + 1)]
    odrs.sort(key=lambda x: (x["unit"], x["floor"]))

    oid2t = {o["id"]: t for o, t in zip(odrs, ts)}
    t = oid2t[odrs[0]["id"]]
    t_stones = [[st + (t - st)/3, ARRANGE], [st + (t - st)*2/3, UP], [t, DELIVER]]

    last_unit = odrs[0]["unit"]
    for odr in odrs[1:]:
        t1 = t_stones[-1][0]
        t2 = oid2t[odr["id"]]
        if odr["unit"] == last_unit:
            t = (t2 - t1) / 2
            t_stones.append([t1 + t, UP])
            t_stones.append([t2, DELIVER])
        else:
            t = (t2 - t1) / 4
            t_stones.append([t1 + t, DOWN])
            t_stones.append([t1 + 2 * t, UNIT])
            t_stones.append([t1 + 3 * t, UP])
            t_stones.append([t2, DELIVER])
            last_unit = odr["unit"]

    last_et = st
    ranges = []
    for t, atp in t_stones:
        ranges.append([[last_et, t], atp])
        last_et = t
    t = (last_et + et) / 2
    ranges.append([[last_et, t], DOWN])
    ranges.append([[t, et], ARRANGE])

    return oid2t, ranges


def fine_rec_smt(st, et, odrs):
    ts = sorted([o["finish_time"] for o in odrs])
    for i in range(len(ts) - 1):
        t1, t2 = ts[i], ts[i+1]
        if t1 >= t2:
            ts[i+1] = t1 + 1e-3
    if len(ts) > 1:
        T = et - st
        sst = st + T / 10
        eet = et - T / 10
        p = (eet - sst) / (ts[-1] - ts[0])
        ofst = sst - ts[0] * p
        ts = [t * p + ofst for t in ts]
    else:
        assert len(ts) == 1
        if not st <= ts[0] <= et:
            ts = [(st + et) / 2]

    odrs.sort(key=lambda x: (x["unit"], x["floor"]))
    oid2t = {o["id"]: t for o, t in zip(odrs, ts)}
    t = oid2t[odrs[0]["id"]]
    # t_stones = [[t/3, ARRANGE], [t*2/3, UP], [t, DELIVER]]
    t_stones = [[st + (t - st)/3, ARRANGE], [st + (t - st)*2/3, UP], [t, DELIVER]]
    last_unit = odrs[0]["unit"]
    for odr in odrs[1:]:
        t1 = t_stones[-1][0]
        t2 = oid2t[odr["id"]]
        if odr["unit"] == last_unit:
            t = (t2 - t1) / 2
            t_stones.append([t1 + t, UP])
            t_stones.append([t2, DELIVER])
        else:
            t = (t2 - t1) / 4
            t_stones.append([t1 + t, DOWN])
            t_stones.append([t1 + 2 * t, UNIT])
            t_stones.append([t1 + 3 * t, UP])
            t_stones.append([t2, DELIVER])
            last_unit = odr["unit"]

    last_et = st
    ranges = []
    for t, atp in t_stones:
        ranges.append([[last_et, t], atp])
        last_et = t
    t = (last_et + et) / 2
    ranges.append([[last_et, t], DOWN])
    ranges.append([[t, et], ARRANGE])

    return oid2t, ranges


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def xavier_init(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class DefaultArgs:
    def __init__(self):
        self.seed = 233
        self.cuda = 2
        self.infer_cuda = -1

        self.valid_ratio = 0.25
        self.batch_size = 100
        self.epoch = 300
        self.early_stop = 30
    
        self.dim_cidx = 8
        self.num_cidx = 3
        self.dim_slot = 8
        self.num_slot = 5


class Model(nn.Module):
    def __init__(self, dim_cidx, num_cidx, dim_slot, num_slot):
        super().__init__()
        self.emb_cidx = nn.Embedding(num_cidx, dim_cidx)
        self.emb_slot = nn.Embedding(num_slot, dim_slot)
        ff_dims = [24, 24, 12, 1]
        layers = [nn.Linear(i, j) for i, j in zip(ff_dims, ff_dims[1:])]
        self.ff = nn.Sequential(
            *sum(
                [[l, nn.ReLU(), nn.Dropout(0.1)] for l in layers[:-1]],
                []
            ),
            layers[-1])

    def forward(self, x):
        cidxs, slots, fs = zip(*x)
        cidxs = self.emb_cidx(torch.stack(cidxs))
        slots = self.emb_slot(torch.stack(slots))
        x = torch.hstack([cidxs, slots, torch.vstack(fs)])
        y = self.ff(x).view(-1)
        return y
    

class DeepFineRec:
    def __init__(self, cid2cidx, args=DefaultArgs()):
        self.args = args
        set_seed(args.seed)
        device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 else "cpu")
        self.device = device
        self.model = None
        self.cid2cidx = cid2cidx

    def dataset_to_tensor(self, dataset, device):
        return [
            (
                [
                    (
                        [
                            torch.tensor(cidx, dtype=torch.int, device=device),
                            torch.tensor(slot, dtype=torch.int, device=device),
                            torch.tensor([is_weekend, arrange, unit, up, down, d, c, b], dtype=torch.float, device=device),
                        ], 
                        torch.tensor(y, dtype=torch.float, device=device)
                    ) for (arrange, unit, up, down, d, c, b, cidx, is_weekend, slot), y in datas
                ],
                torch.tensor(ty, dtype=torch.float, device=device)
            ) for datas, ty in dataset
        ]

    def load_model(self, path):
        device = torch.device(f"cuda:{self.args.infer_cuda}" if self.args.infer_cuda >= 0 else "cpu")
        model = Model(
            dim_cidx=self.args.dim_cidx,
            num_cidx=self.args.num_cidx,
            dim_slot=self.args.dim_slot,
            num_slot=self.args.num_slot
        ).to(device)
        model.load_state_dict(torch.load(path))
        self.model = model

    def train(self, train_data, test_data, cid2params):
        run_name = time.strftime("DeepFineRec_%y%m%d_%H%M%S")
        setproctitle(f"{run_name}@yufudan")
        os.makedirs(f"log/{run_name}")

        def get_dataset(wave, params, cidx):
            date = wave["date"]
            date = date - 800 if date > 800 else date + 92
            t = date % 7
            if t == 0:
                t = 7
            is_weekend = t > 5
            slot = int((np.mean(wave["wave_traj"]) - 8 * 3600) / 3 / 3600)
            slot = max(0, min(4, slot))

            datas = []
            oid2odr = {o["id"]: o for o in wave["orders"]}
            for s in wave["stays"]:
                if not s["oids_matched"]:
                    continue
                odrs = [oid2odr[oid] for oid in s["oids_matched"]]
                st, et = s["trange"]
                t_stones = fine_rec_abm(st, et, odrs, params)[1]
                ts, atps = zip(*t_stones)
                ts = [st, *ts[:-1]] 
                tos = [t for t, atp in zip(ts, atps) if atp == DELIVER]
                tfirst = tos[0] - st
                tmids = [t2 - t1 for t1, t2 in zip(tos, tos[1:])]
                tlast = et - tos[-1]

                unit2odrs = defaultdict(list)
                for odr in odrs:
                    unit2odrs[odr["unit"]].append(odr)
                ufdcbs = []
                for i, odrs in enumerate(unit2odrs.values()):
                    floor2odrs = defaultdict(list)
                    for odr in odrs:
                        floor2odrs[odr["floor"]].append(odr)
                    for f, odrs in sorted(list(floor2odrs.items()), key=lambda x:x[0]):
                        t = Counter([o["type"] for o in odrs])
                        ufdcbs.append([i, f-1, t[ORDER_DELIVER], t[ORDER_CPICK], t[ORDER_BPICK]])
                assert len(ufdcbs) == len(tmids) + 1

                # arrange, unit, up, down, d, c, b; cidx, is_weekend, slot
                data = [
                    [[1, 0, ufdcbs[0][1], 0, 0, 0, 0, cidx, is_weekend, slot], tfirst],
                    [[1, 0, 0, *ufdcbs[-1][1:], cidx, is_weekend, slot], tlast],
                ]
                for (u1, f1, *dcb1), (u2, f2, *_), tmid in zip(ufdcbs, ufdcbs[1:], tmids):
                    data.append([[
                        0, int(u1 != u2), 
                        f2 - f1 if u1 == u2 else f2, 
                        0 if u1 == u2 else f1, 
                        *dcb1, cidx, is_weekend, slot], tmid])
                datas.append([data, et - st])
            return datas
        
        cid2idx = self.cid2cidx
        train_set = []
        for w in tqdm(train_data):
            train_set += get_dataset(w, cid2params[w["cid"]], cid2idx[w["cid"]])
        test_set = []
        for w in tqdm(test_data):
            test_set += get_dataset(w, cid2params[w["cid"]], cid2idx[w["cid"]])
        print("train:", len(train_set), "test:", len(test_set))

        device = self.device
        train_set = self.dataset_to_tensor(train_set, device)
        test_set = self.dataset_to_tensor(test_set, device)
        if self.args.valid_ratio > 0:
            n_valid = int(self.args.valid_ratio * len(train_set))
            random.shuffle(train_set)
            valid_set = train_set[:n_valid]
            train_set = train_set[n_valid:]
        else:
            valid_set = []
        batch_size = min(len(train_set), self.args.batch_size)
        print("train, valid, test:", len(train_set), len(valid_set), len(test_set))

        model = Model(
            dim_cidx=self.args.dim_cidx,
            num_cidx=self.args.num_cidx,
            dim_slot=self.args.dim_slot,
            num_slot=self.args.num_slot
        ).to(device)
        model.apply(xavier_init)
        opt = torch.optim.Adam(model.parameters())
        criterion = nn.L1Loss(reduction='sum')
        
        best_epoch = -1
        best_valid_loss = best_test_loss = float("Inf")
        valid_loss = test_loss = -1
        with tqdm(range(self.args.epoch), dynamic_ncols=True) as bar: 
            for epoch in bar:
                n = 0
                train_loss = 0
                random.shuffle(train_set)
                for i in range(0, len(train_set) // batch_size * batch_size, batch_size):
                    data = train_set[i : i + batch_size]
                    events, tys = zip(*data)
                    nes = [len(e) for e in events]
                    x, y = zip(*[(x, y) for e in events for x, y in e])
                    y = torch.stack(y)
                    y_pred = model(x)

                    presum = [0] * (len(nes) + 1)
                    for i, ne in enumerate(nes):
                        presum[i+1] = presum[i] + ne
                    loss = torch.stack([
                        criterion(y_pred[s:e] / y_pred[s:e].sum() * ty, y[s:e])
                        for s, e, ty in zip(presum, presum[1:], tys)
                    ]).sum() / len(x)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    train_loss += loss.cpu().item()
                    n += 1
                    bar.set_description(
                        f"train: {train_loss/n:.4f} " + 
                        f"valid: {valid_loss:.4f} " +
                        f"test:  {test_loss:.4f} ")

                with torch.no_grad():
                    events, tys = zip(*test_set)
                    nes = [len(e) for e in events]
                    x, y = zip(*[(x, y) for e in events for x, y in e])
                    y = torch.stack(y)
                    y_pred = model(x)
                    presum = [0] * (len(nes) + 1)
                    for i, ne in enumerate(nes):
                        presum[i+1] = presum[i] + ne
                    test_loss = torch.stack([
                        criterion(y_pred[s:e] / y_pred[s:e].sum() * ty, y[s:e])
                        for s, e, ty in zip(presum, presum[1:], tys)
                    ]).sum().cpu().item() / len(x)
                    if len(valid_set) > 0:
                        events, tys = zip(*valid_set)
                        nes = [len(e) for e in events]
                        x, y = zip(*[(x, y) for e in events for x, y in e])
                        y = torch.stack(y)
                        y_pred = model(x)
                        presum = [0] * (len(nes) + 1)
                        for i, ne in enumerate(nes):
                            presum[i+1] = presum[i] + ne
                        valid_loss = torch.stack([
                            criterion(y_pred[s:e] / y_pred[s:e].sum() * ty, y[s:e])
                            for s, e, ty in zip(presum, presum[1:], tys)
                        ]).sum().cpu().item() / len(x)
                    else:
                        valid_loss = test_loss

                if epoch > 10 and valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_test_loss =  test_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), f"log/{run_name}/{epoch}.pt")

                if epoch > 10 and epoch - best_epoch > self.args.early_stop:
                    print(f"early stopped")
                    break
            print("best epoch:", best_epoch, "valid loss:", best_valid_loss, "test acc:", best_test_loss)

    def infer(self, st, et, odrs, cid, is_weekend, slot):
        device = torch.device(f"cuda:{self.args.infer_cuda}" if self.args.infer_cuda >= 0 else "cpu")
        model = self.model
        assert model is not None

        cidx = self.cid2cidx[cid]
        unit2odrs = defaultdict(list)
        for odr in odrs:
            unit2odrs[odr["unit"]].append(odr)
        ufdcbos = []
        for i, odrs in enumerate(unit2odrs.values()):
            floor2odrs = defaultdict(list)
            for odr in odrs:
                floor2odrs[odr["floor"]].append(odr)
            for f, odrs in sorted(list(floor2odrs.items()), key=lambda x:x[0]):
                t = Counter([o["type"] for o in odrs])
                ufdcbos.append([i, f-1, t[ORDER_DELIVER], t[ORDER_CPICK], t[ORDER_BPICK], odrs])
        # cidxs, slots, fs = zip(*x)  # f: is_weekend, arrange, unit, up, down, d, c, b
        x = [[1, 0, ufdcbos[0][1], 0, 0, 0, 0, cidx, is_weekend, slot]]
        for (u1, f1, *dcb1, _), (u2, f2, *_) in zip(ufdcbos, ufdcbos[1:]):
            x.append([
                0, int(u1 != u2), 
                f2 - f1 if u1 == u2 else f2, 
                0 if u1 == u2 else f1, 
                *dcb1, cidx, is_weekend, slot])
        x.append([1, 0, 0, *ufdcbos[-1][1:-1], cidx, is_weekend, slot])
        x = [
            [
                torch.tensor(cidx, dtype=torch.int, device=device),
                torch.tensor(slot, dtype=torch.int, device=device),
                torch.tensor([is_weekend, arrange, unit, up, down, d, c, b], dtype=torch.float, device=device),
            ] for arrange, unit, up, down, d, c, b, cidx, is_weekend, slot in x
        ]
        y = model(x)
        y = (y / y.sum() * (et - st)).tolist()

        oid2t = {}
        t_stones = []
        if ufdcbos[0][1] > 0:
            t_stones += [[y[0] / 2, ARRANGE], [y[0], UP]]
        else:
            t_stones.append([y[0], ARRANGE])

        t = y[0]
        for (u1, f1, *_, odrs1), (u2, f2, *_), yy in zip(ufdcbos, ufdcbos[1:], y[1:-1]):
            if u1 == u2:
                atps = [DELIVER, UP]
            else:
                atps = [DELIVER]
                if f1 > 0:
                    atps.append(DOWN)
                atps.append(UNIT)
                if f2 > 0:
                    atps.append(UP)
            for atp in atps:
                t += yy / len(atps)
                t_stones.append([t, atp])
                if atp == DELIVER:
                    for o in odrs1:
                        oid2t[o["id"]] = t

        f, *_, odrs = ufdcbos[-1][1:]
        if f > 0:
            atps = [DELIVER, DOWN, ARRANGE]
        else:
            atps = [DELIVER, ARRANGE]
        tt = (et - st - t) / len(atps)
        for atp in atps:
            t += tt
            t_stones.append([t, atp])
            if atp == DELIVER:
                for o in odrs:
                    oid2t[o["id"]] = t
        
        oid2t = {oid: t + st for oid, t in oid2t.items()}
        t_stones = [[t + st, atp] for t, atp in t_stones]
        last_et = st
        ranges = []
        for t, atp in t_stones:
            ranges.append([[last_et, t], atp])
            last_et = t

        return oid2t, ranges
