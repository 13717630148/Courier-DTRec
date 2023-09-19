import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from setproctitle import setproctitle
from shapely.geometry import LineString, MultiPoint, Point
from tqdm import tqdm

from constants import ORDER_BPICK, ORDER_CPICK, ORDER_DELIVER, buildings, group_by

random.seed(233)

MAX_T_PREV = 120
MAX_T_POST = 1800
MAX_DIS = 100
MIN_MATCH_SCORE = 0.1 
BIG_MATCH_SCORE = 0.7 
SCORE_INSTAY = 1.2
TRAIN_DAYS = list(range(801, 832)) + list(range(901, 906))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DefaultArgs:
    def __init__(self):
        self.seed = 233
        self.cuda = 1

        self.valid_ratio = 0
        self.batch_size = 100
        self.epoch = 100
        self.early_stop = 10 
        
        self.dim_cidx = 8
        self.num_cidx = 3
        self.dim_slot = 8
        self.num_slot = 5


def cal_score_t(ts1, ts2, to):
    if to < ts1:
        t = ts1 - to
        return 0 if t > MAX_T_PREV else 1 - t / MAX_T_PREV
    elif to > ts2:
        t = to - ts2
        if t > MAX_T_POST:
            return 0
        else:
            return 1 - t / MAX_T_POST
    else:
        return SCORE_INSTAY
    

def cal_score_s(xs, ys, xo, yo):
    dis = ((xo - xs) ** 2 + (yo - ys) ** 2) ** 0.5
    return 0 if dis > MAX_DIS else 1 - dis / MAX_DIS


def match_stay_order(wave, adr_ref, method):
    stays, orders = wave["stays"], wave["orders"]
    if not stays or not orders:
        return wave
    
    for s in stays:
        s["oids_matched"] = []
        s["match_xys"] = []
        s["match_scores"] = []
    for o in orders:
        oid = o["id"]
        t = o["finish_time"]
        scores_t = [1] * len(stays) if method == "s" else [cal_score_t(*s["trange"], t) for s in stays]
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
        has_big_score = False
        for xy in o["xys"]:
            scores = []
            for s, score_t in zip(stays, scores_t): 
                if score_t < MIN_MATCH_SCORE:
                    scores.append(0)
                else:
                    score_s = 1 if method == "t" else cal_score_s(*s["point"][:2], *xy)
                    scores.append(score_t * score_s)
            max_score = max(scores)
            idxs = [i for i, score in enumerate(scores) if score == max_score]
            idx = idxs[0] if len(idxs) == 1 else random.choice(idxs)
            match_score = max_score
            if max_score > BIG_MATCH_SCORE:
                o["sid_matched"] = idx
                o["match_xy"] = xy
                o["match_score"] = match_score
                stays[idx]["oids_matched"].append(oid)
                stays[idx]["match_xys"].append(xy)
                stays[idx]["match_scores"].append(match_score)
                has_big_score = True
                break
            elif max_score >= MIN_MATCH_SCORE:
                matched_results.append([idx, xy, match_score])
        if not has_big_score and matched_results:
            max_score = max(x[-1] for x in matched_results)
            tmp = [x for x in matched_results if x[-1] == max_score]
            idx, xy, score = tmp[0] if len(tmp) == 1 else random.choice(tmp)
            o["sid_matched"] = idx
            o["match_xy"] = xy
            o["match_score"] = score
            stays[idx]["oids_matched"].append(oid)
            stays[idx]["match_xys"].append(xy)
            stays[idx]["match_scores"].append(score)

    return wave


def coarse_rec(wave_data, adr_ref, method):
    if method in ["s", "t", "st"]:
        for w in tqdm(wave_data):
            match_stay_order(w, adr_ref, method)
        return wave_data
    
    train_data = [x for x in wave_data if x["date"] in TRAIN_DAYS]
    test_data = [x for x in wave_data if x["date"] not in TRAIN_DAYS]
    metastp = metaSTP(train_data=train_data, test_data=test_data)
    metastp.train()
    match_results = metastp.infer(model_path="log/metaSTP_230608_163624/40.pt")
    wid2w = {(w["cid"], w["date"], w["wave_idx"]): w for w in test_data}
    wid2pairs = defaultdict(list)
    for wid, oids, sid in match_results:
        wid2pairs[tuple(wid)].append([oids, sid])
    for wid, pairs in wid2pairs.items():
        wave = wid2w[wid]
        stays = wave["stays"]
        for s in stays:
            s["oids_matched"] = []
            s["match_xys"] = []
            s["match_scores"] = []
        oid2odr = {o["id"]: o for o in wave["orders"]}
        for oids, sid in pairs:
            s = stays[sid]
            s["oids_matched"] = oids
            s["match_xys"] = [[-1, -1] for _ in range(len(oids))]
            s["match_scores"] = [-1] * len(oids)
            for oid in oids:
                o = oid2odr[oid]
                o["sid_matched"] = sid
                o["match_xy"] = [-1, -1]
                o["match_score"] = -1

    return wave_data


class Model(nn.Module):
    def __init__(self, dim_cidx, num_cidx, dim_slot, num_slot):
        super().__init__()
        self.emb_cidx = nn.Embedding(num_cidx, dim_cidx)
        self.emb_slot = nn.Embedding(num_slot, dim_slot)
        self.emb_floor = nn.Linear(4, 8)
        self.fuse_floor = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=8, nhead=4, dim_feedforward=16),
            num_layers=3)
        self.emb_unit = nn.Sequential(nn.Linear(8, 16), nn.ReLU())

        ff_dims = [42, 20, 1]
        layers = [nn.Linear(i, j) for i, j in zip(ff_dims, ff_dims[1:])]
        self.ff_match = nn.Sequential(
            *sum(
                [[l, nn.ReLU(), nn.Dropout(0.1)] for l in layers[:-1]],
                []
            ),
            layers[-1])

    def forward(self, x):
        f1, f2, f3, nf3 = zip(*x)
        cidxs, slots, gfeats = zip(*f1)
        cidxs = self.emb_cidx(torch.stack(cidxs))
        slots = self.emb_slot(torch.stack(slots))
        gfeats = torch.vstack([
            self.emb_unit(
                torch.vstack([
                    self.fuse_floor(self.emb_floor(ufeat)).sum(dim=0)
                    for ufeat in gfeat
                ])
            ).sum(dim=0)
            for gfeat in gfeats
        ])
        f1f2 = torch.hstack([cidxs, slots, gfeats, torch.vstack(f2)])
        nf3 = torch.stack(nf3)
        f1f2 = torch.repeat_interleave(f1f2, repeats=nf3, dim=0)
        x = torch.hstack([f1f2, torch.vstack(f3)])
        y = self.ff_match(x).view(-1)
        return y, nf3


class metaSTP:
    def __init__(self, train_data, test_data, args=DefaultArgs()):
        self.args = args
        set_seed(args.seed)
        device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 else "cpu")
        self.device = device

        def get_dataset(wave, cidx, wid):
            stays = wave["stays"]
            oid2odr = {o["id"]: o for o in wave["orders"]}
            groups = [[oid2odr[oid] for oid in s["oids_matched"]] for s in stays if s["oids_matched"]]
            candidates = []
            for odrs in groups:
                sid2scores = defaultdict(list)
                for o in odrs:
                    t = o["finish_time"]
                    xys = o["xys"]
                    for sid, s in enumerate(stays):
                        score_t = cal_score_t(*s["trange"], t)
                        if score_t > MIN_MATCH_SCORE:
                            score = score_t * max(cal_score_s(*s["point"][:2], *xy) for xy in xys)
                            if score > MIN_MATCH_SCORE:
                                sid2scores[sid].append(score)
                sid_score = [(sid, max(scores)) for sid, scores in sid2scores.items()]
                sid_score.sort(key=lambda x:-x[1])
                sids = [x[0] for x in sid_score[:10]]

                true_sid = odrs[0]["sid_matched"]
                for o in odrs[1:]:
                    assert o["sid_matched"] == true_sid
                if true_sid in sids:
                    sids = [true_sid] + [x for x in sids if x != true_sid]
                else:
                    sids = [true_sid] + sids[:9]
                candidates.append(sids)
            
            sids_all = {sid for sids in candidates for sid in sids}
            sid2feature = {}
            for sid in sids_all:
                s = stays[sid]
                ps = s["traj"]
                ts, te = s["trange"]
                poly = MultiPoint([p[:2] for p in ps]).convex_hull
                if isinstance(poly, LineString):
                    length = poly.length
                    area = 0
                elif isinstance(poly, Point):
                    length = 0
                    area = 0
                else:
                    area = poly.area
                    length = poly.length
                sid2feature[sid] = [te - ts, area, length]
            
            group_features = []
            for odrs in groups:
                feature = []
                tu, tf, td, tc, tb = 0, 0, 0, 0, 0
                for uodrs in group_by(odrs, "unit").values():
                    tu += 1
                    u_features = []
                    for f, fodrs in group_by(uodrs, "floor").items():
                        tf += 1
                        t = Counter(o["type"] for o in fodrs)
                        d, c, b = t[ORDER_DELIVER], t[ORDER_CPICK], t[ORDER_BPICK]
                        td, tc, tb = td + d, tc + c, tb + b
                        u_features.append([f, d, c, b])
                    feature.append(sorted(u_features, key=lambda x: x[0]))
                total_feature = [tu, tf, td, tc, tb]
                group_features.append([feature, total_feature])

            def get_wfeature(wave, cidx):
                date = wave["date"]
                date = date - 800 if date > 800 else date + 92
                t = date % 7
                if t == 0:
                    t = 7
                is_weekend = t > 5
                slot = int((np.mean(wave["wave_traj"]) - 8 * 3600) / 3 / 3600)
                slot = max(0, min(4, slot))
                return [cidx, is_weekend, slot]

            cidx, is_weekend, slot = get_wfeature(wave, cidx)
            dataset = []
            for odrs, (gfeat, tgfeat), sids in zip(groups, group_features, candidates):
                xys = [o["match_xy"] for o in odrs]
                oids = [o["id"] for o in odrs]
                sfeats = []
                for sid in sids:
                    sx, sy = stays[sid]["point"][:2]
                    dis = np.mean([((sx - x)**2 + (sy - y)**2)**0.5 for x, y in xys])
                    sfeats.append([*sid2feature[sid], dis])
                dataset.append([
                    [cidx, slot, gfeat],
                    [is_weekend, tgfeat],
                    sfeats,               
                    0,                     
                    [wid, oids, sids]      
                ])
            return dataset

        cids = list(set(w["cid"] for ws in [train_data, test_data] for w in ws))
        cid2idx = {cid: i for i, cid in enumerate(sorted(cids))}
        train_set = []
        for w in tqdm(train_data):
            wid = [w["cid"], w["date"], w["wave_idx"]]
            train_set += get_dataset(w, cid2idx[w["cid"]], wid)
        test_set = []
        for w in tqdm(test_data):
            wid = [w["cid"], w["date"], w["wave_idx"]]
            test_set += get_dataset(w, cid2idx[w["cid"]], wid)

        print("train, test:", len(train_set), len(test_set))
        self.train_set = train_set
        self.test_set = test_set

    def dataset_to_tensor(self, dataset, device, nontrivial_only=True):
        return [
            [
                [
                    torch.tensor(cidx, dtype=torch.int, device=device),
                    torch.tensor(slot, dtype=torch.int, device=device),
                    [torch.tensor(ufeat, dtype=torch.float, device=device) for ufeat in gfeat]
                ],
                torch.tensor([is_weekend, *tgfeat], dtype=torch.float, device=device),
                torch.tensor(sfeats, dtype=torch.float, device=device),
                torch.tensor(len(sfeats), dtype=torch.int, device=device),
                torch.tensor(y, dtype=torch.long, device=device),
                others
            ] for (cidx, slot, gfeat), (is_weekend, tgfeat), sfeats, y, others in dataset \
            if not nontrivial_only or len(sfeats) > 1
        ]

    def train(self):
        run_name = time.strftime("metaSTP_%y%m%d_%H%M%S")
        setproctitle(f"{run_name}@yufudan")
        os.makedirs(f"log/{run_name}")

        device = self.device
        train_set = self.dataset_to_tensor(self.train_set, device, nontrivial_only=True)
        test_set = self.dataset_to_tensor(self.test_set, device, nontrivial_only=True)
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
        opt = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        best_epoch = best_valid_acc = best_test_acc = -1
        valid_loss = valid_acc = test_loss = test_acc = -1
        with tqdm(range(self.args.epoch), dynamic_ncols=True) as bar: 
            for epoch in bar:
                n = 0
                train_loss = train_acc = 0
                random.shuffle(train_set)
                for i in range(0, len(train_set) // batch_size * batch_size, batch_size):
                    data = train_set[i : i + batch_size]
                    x = [d[:4] for d in data]
                    y = [d[4] for d in data]
                    y = torch.stack(y)
                    y_pred, ncs = model(x)
                    ncs = ncs.cpu()
                    presum = [0] * (len(ncs) + 1)
                    for i, nc in enumerate(ncs):
                        presum[i+1] = presum[i] + nc
                    loss = torch.stack([
                        criterion(y_pred[s:e], yy)
                        for yy, s, e in zip(y, presum, presum[1:])
                    ]).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    train_loss += loss.cpu().item()
                    train_acc += torch.stack([
                        y_pred[s:e].argmax() == yy
                        for yy, s, e in zip(y, presum, presum[1:])
                    ]).float().mean().cpu().item()
                    n += 1
                    bar.set_description(
                        f"train: {train_loss/n:.4f} {train_acc/n*100:.2f}% " + 
                        f"valid: {valid_loss:.4f} {valid_acc*100:.2f}% " +
                        f"test:  {test_loss:.4f} {test_acc*100:.2f}% ")

                with torch.no_grad():
                    x = [d[:4] for d in test_set]
                    y = [d[4] for d in test_set]
                    y = torch.stack(y)
                    y_pred, ncs = model(x)
                    ncs = ncs.cpu()
                    presum = [0] * (len(ncs) + 1)
                    for i, nc in enumerate(ncs):
                        presum[i+1] = presum[i] + nc
                    test_loss = torch.stack([
                        criterion(y_pred[s:e], yy)
                        for yy, s, e in zip(y, presum, presum[1:])
                    ]).mean().cpu().item()
                    test_acc = torch.stack([
                        y_pred[s:e].argmax() == yy
                        for yy, s, e in zip(y, presum, presum[1:])
                    ]).float().mean().cpu().item()

                    if len(valid_set) > 0:
                        x = [d[:4] for d in valid_set]
                        y = [d[4] for d in valid_set]
                        y = torch.stack(y)
                        y_pred, ncs = model(x)
                        presum = [0] * (len(ncs) + 1)
                        for i, nc in enumerate(ncs):
                            presum[i+1] = presum[i] + nc
                        valid_loss = torch.stack([
                            criterion(y_pred[s:e], yy)
                            for yy, s, e in zip(y, presum, presum[1:])
                        ]).mean().cpu().item()
                        valid_acc = torch.stack([
                            y_pred[s:e].argmax() == yy
                            for yy, s, e in zip(y, presum, presum[1:])
                        ]).float().mean().cpu().item()
                    else:
                        valid_loss, valid_acc = test_loss, test_acc

                if epoch > 10 and valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_test_acc =  test_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), f"log/{run_name}/{epoch}.pt")

                if epoch > 10 and epoch - best_epoch > self.args.early_stop:
                    print(f"early stopped")
                    print("best epoch:", best_epoch, "valid acc:", best_valid_acc, "test acc:", best_test_acc)
                    break

    def infer(self, model_path):
        device = self.device
        test_set = self.dataset_to_tensor(self.test_set, device, nontrivial_only=False)
        model = Model(
            dim_cidx=self.args.dim_cidx,
            num_cidx=self.args.num_cidx,
            dim_slot=self.args.dim_slot,
            num_slot=self.args.num_slot
        ).to(device)
        model.load_state_dict(torch.load(model_path))

        with torch.no_grad():
            x = [d[:4] for d in test_set]
            y = [d[4] for d in test_set]
            y = torch.stack(y)
            y_pred, ncs = model(x)
            ncs = ncs.cpu()
            presum = [0] * (len(ncs) + 1)
            for i, nc in enumerate(ncs):
                presum[i+1] = presum[i] + nc
            acc = 0
            yys_pred = []
            for yy, s, e in zip(y, presum, presum[1:]):
                yy_pred = y_pred[s:e].argmax()
                acc += yy_pred == yy
                yys_pred.append(yy_pred)
            print("acc:", (acc / len(test_set)).item())

            results = []
            for data, yy_pred in zip(test_set, yys_pred):
                wid, oids, sids = data[-1]
                results.append([wid, oids, sids[yy_pred]])
        
        return results
    
