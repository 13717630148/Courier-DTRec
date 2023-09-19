import pickle
from collections import defaultdict

import folium
from coord_convert.transform import wgs2gcj
from pyproj import Proj
from shapely.geometry import Polygon

projector = Proj("+proj=tmerc +lat_0=39.816344 +lon_0=116.520481")
LON_CEN, LAT_CEN = 116.520481, 39.816344
X_CEN, Y_CEN = projector(LON_CEN, LAT_CEN)

ORDER_DELIVER = "deliver"
ORDER_CPICK = "cpick"
ORDER_BPICK = "bpick"

WORK = "work"    
REST = "rest" 
IGNORE = "ignore"
OTHER = "other"
UP = 1
DOWN = 2
UNIT = 3
DELIVER = 4
ARRANGE = 5
NOT_WORK = 6

buildings = pickle.load(open("dataset/buildings.pkl", "rb"))
for b in buildings:
    b["gate_xy"] = projector(*b["gate_gps"])
    b["poly"] = Polygon([projector(*p) for p in b["points"]])
buildings = {bd["id"]: bd for bd in buildings}

regions = pickle.load(open("dataset/regions.pkl", "rb"))
for r in regions:
    r["poly"] = Polygon([projector(*p) for p in r["boundary"]])
regions = {r["id"]: r for r in regions}

LON_STA, LAT_STA = 116.516869, 39.808934
X_STA, Y_STA = projector(LON_STA, LAT_STA)


def time_conventer(t):
    t = round(t)
    assert 0 <= t < 86400
    h = t // 3600
    t -= h * 3600
    m = t // 60
    s = t - m * 60
    h = str(h) if h > 9 else f"0{h}"
    m = str(m) if m > 9 else f"0{m}"
    s = str(s) if s > 9 else f"0{s}"
    return ":".join([h, m, s])


def get_base_map():
    m = folium.Map(
        location=[LAT_CEN, LON_CEN],
        control_scale=True,
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='gaodemap',
        zoom_start=20,
    )
    for b in buildings.values():
        color = "orange" if b["is_elevator"] else "black"
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
            opacity=0.8,
            weight=0.5,
            color=color,
        ).add_to(m)
    return m


def xy2loc(xy):
    return wgs2gcj(*projector(*xy, inverse=True))[::-1]


default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def print_table(columns, lines):
    def mylen(s):
        return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in s)
    lens = [max(15, mylen(k) + 3) for k in columns]
    head = "".join(k + " " * (l-mylen(k)) for k, l in zip(columns, lens))
    print(head)
    print("=" * (mylen(head) - 3))
    for line in lines:
        line = [f"{x:.4f}" if not isinstance(x, str) else x for x in line]
        print("".join(x + " "*(l - mylen(x)) for x, l in zip(line, lens)))


def group_by(arr, key):
    if len(arr) == 0:
        return {}
    assert isinstance(arr[0], dict)
    r = defaultdict(list)
    if isinstance(key, str):
        for a in arr:
            r[a[key]].append(a)
    else:
        assert isinstance(key, list)
        for a in arr:
            r[tuple(a[k] for k in key)].append(a)
    return r
