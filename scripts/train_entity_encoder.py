"""Train the GeoAnchor spatial entity encoder.

The encoder combines 38 explicit features with 218 learned relational
dimensions.

  Inputs:
    data/source/osm_rdf/graph_entity_index.json
    data/source/osm_rdf/relation_index.json
    data/source/osm_rdf/triples_train.tsv
    data/source/osm_rdf/entities.parquet
    data/source/osm_rdf/geometry.parquet
    data/source/osm_rdf/entity_features.npy

  Losses:
    L_TransE  (margin-ranking)
    L_prox    (proximity:  triplet  d_emb(a,p) < d_emb(a,q)  when d_geo(a,p) < d_geo(a,q))
    L_scale   (admin clustering: cos(a,b) == 1 iff same admin_level)
    L_topo    (sf*/parent/child relations: |h - t| small)
    L_relpos  (relative-position analogy via cos vs exp(-dist))
    L_type    (NEW: cos(a,b) == 1 iff same osm_type)
    L_density (NEW: cos(a,b) high when POI density (vec[21]) similar)

  Outputs (out_dir):
    best_entity_embeddings.pt
    best_encoder.pt
    entity_embeddings.pt
    train_log.jsonl
    feature_index.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genanchor.entity_encoder import GeoKGEModel

# ----------- utilities -----------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = torch.deg2rad(lat1); p2 = torch.deg2rad(lat2)
    dp = p2 - p1
    dl = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dp/2)**2 + torch.cos(p1) * torch.cos(p2) * torch.sin(dl/2)**2
    return 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))


def load_triple_ids(path, e2i, r2i):
    print(f"[load] {path}", flush=True)
    t0 = time.time()
    h_, r_, t_ = [], [], []
    drop = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3: drop += 1; continue
            h, r, t = parts
            if h in e2i and r in r2i and t in e2i:
                h_.append(e2i[h]); r_.append(r2i[r]); t_.append(e2i[t])
            else:
                drop += 1
    arr = np.stack([np.array(h_, dtype=np.int32),
                    np.array(r_, dtype=np.int32),
                    np.array(t_, dtype=np.int32)], axis=1)
    print(f"  kept {len(arr):,} triples (dropped {drop:,}) [{time.time()-t0:.1f}s]", flush=True)
    return arr


class TripleDS(Dataset):
    def __init__(self, arr): self.arr = arr
    def __len__(self): return len(self.arr)
    def __getitem__(self, i): return self.arr[i]


def collate(batch):
    return torch.from_numpy(np.stack(batch, 0)).long()


# ----------- auxiliary feature builder -----------

# Tiny geohash encoder (same as f1_2)
B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
def geohash_encode(lat: float, lon: float, precision: int) -> str:
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    bits, bit, ch = [], 0, 0
    even = True
    while len(bits) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon > mid: ch |= (1 << (4 - bit)); lon_lo = mid
            else: lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat > mid: ch |= (1 << (4 - bit)); lat_lo = mid
            else: lat_hi = mid
        even = not even; bit += 1
        if bit == 5:
            bits.append(B32[ch]); bit, ch = 0, 0
    return "".join(bits)


def build_aux_features(e2i: dict, df_ent: pd.DataFrame):
    """Return dict of np.arrays aligned to entity id order:
       lon, lat, has_geom, gh4_str, gh5_str, gh6_str, admin_level (int), osm_type_idx, entity_iri
    """
    n = len(e2i)
    # map IRI → row in df_ent
    ent_to_row = {iri: i for i, iri in enumerate(df_ent["entity"].values)}

    lon = np.zeros(n, dtype=np.float32)
    lat = np.zeros(n, dtype=np.float32)
    has_geom = np.zeros(n, dtype=np.float32)
    admin_int = np.full(n, -1, dtype=np.int64)
    osm_type_str = np.full(n, "", dtype=object)
    gh4 = np.full(n, "", dtype=object)
    gh5 = np.full(n, "", dtype=object)
    gh6 = np.full(n, "", dtype=object)
    iris = np.full(n, "", dtype=object)

    df_lon = df_ent["lon"].values
    df_lat = df_ent["lat"].values
    df_admin = df_ent["admin_level"].astype(str).values
    df_osm = df_ent["osm_type"].astype(str).values
    df_iri = df_ent["entity"].values

    miss = 0
    for iri, eid in e2i.items():
        r = ent_to_row.get(iri)
        if r is None:
            miss += 1
            continue
        lo = df_lon[r]
        la = df_lat[r]
        iris[eid] = iri
        osm_type_str[eid] = df_osm[r]
        a_s = df_admin[r]
        if isinstance(a_s, str) and a_s.isdigit():
            admin_int[eid] = int(a_s)
        elif isinstance(a_s, (int, np.integer)):
            admin_int[eid] = int(a_s)
        elif isinstance(a_s, (float, np.floating)) and not (np.isnan(a_s)):
            admin_int[eid] = int(a_s)
        if not (pd.isna(lo) or pd.isna(la)):
            lon[eid] = float(lo); lat[eid] = float(la)
            has_geom[eid] = 1.0
            gh4[eid] = geohash_encode(float(la), float(lo), 4)
            gh5[eid] = geohash_encode(float(la), float(lo), 5)
            gh6[eid] = geohash_encode(float(la), float(lo), 6)
    print(f"  aux: n_geom={int(has_geom.sum()):,}  n_admin={int((admin_int>=0).sum()):,}  miss_iri={miss:,}", flush=True)
    return {
        "lon": lon, "lat": lat, "has_geom": has_geom,
        "admin_int": admin_int, "osm_type_str": osm_type_str,
        "gh4_str": gh4, "gh5_str": gh5, "gh6_str": gh6,
        "entity_iri": iris,
    }


# ----------- diagnostics -----------

def diag_cross_admin_recall(emb, lat, lon, has_geom, par_idx, n=120, radius_km=10.0, topk=20, device="cuda"):
    geom_idx = np.where(has_geom > 0)[0]
    cands = [i for i in geom_idx if par_idx[i] >= 0]
    rng = np.random.default_rng(0); rng.shuffle(cands); cands = cands[:n]
    if not cands: return 0.0
    anchors = torch.tensor(cands, dtype=torch.long, device=device)
    g_t = torch.tensor(geom_idx, dtype=torch.long, device=device)
    lat_t = torch.tensor(lat, dtype=torch.float32, device=device)
    lon_t = torch.tensor(lon, dtype=torch.float32, device=device)
    a_lat = lat_t[anchors]; a_lon = lon_t[anchors]
    chunk = 50000
    gt_sets = [set() for _ in range(len(cands))]
    for s in range(0, len(g_t), chunk):
        e_ = min(len(g_t), s + chunk)
        gl = lat_t[g_t[s:e_]]; gn = lon_t[g_t[s:e_]]
        d = 2*6371.0*torch.asin(torch.clamp(torch.sqrt(
            torch.sin((torch.deg2rad(gl).unsqueeze(0)-torch.deg2rad(a_lat).unsqueeze(1))/2)**2
            + torch.cos(torch.deg2rad(a_lat)).unsqueeze(1)*torch.cos(torch.deg2rad(gl)).unsqueeze(0)
              * torch.sin((torch.deg2rad(gn).unsqueeze(0)-torch.deg2rad(a_lon).unsqueeze(1))/2)**2
        ), 0, 1))
        mask = d < radius_km
        for i in range(len(cands)):
            hits = torch.nonzero(mask[i], as_tuple=True)[0].cpu().numpy()
            for h in hits:
                gid = int(g_t[s + h].item()); aid = int(anchors[i].item())
                if gid == aid: continue
                if par_idx[gid] != par_idx[aid] and par_idx[gid] >= 0 and par_idx[aid] >= 0:
                    gt_sets[i].add(gid)
    q = emb[anchors.cpu()].to(device).float()
    tgt = emb[g_t.cpu()].to(device).float()
    qn = (q ** 2).sum(-1, keepdim=True)
    best_d = None; best_i = None
    chunk = 16384
    for s in range(0, tgt.size(0), chunk):
        e_ = min(tgt.size(0), s + chunk); sub = tgt[s:e_]
        sn = (sub ** 2).sum(-1, keepdim=True).T
        d2 = qn + sn - 2.0 * (q @ sub.T); d2 = torch.clamp(d2, min=0)
        cur_d, cur_i = torch.topk(d2, k=topk+1, dim=-1, largest=False); cur_i = cur_i + s
        if best_d is None: best_d, best_i = cur_d, cur_i
        else:
            cat_d = torch.cat([best_d, cur_d], dim=-1); cat_i = torch.cat([best_i, cur_i], dim=-1)
            best_d, ord_ = torch.topk(cat_d, k=topk+1, dim=-1, largest=False)
            best_i = torch.gather(cat_i, dim=-1, index=ord_)
    topk_local = best_i.cpu().numpy()
    geom_np = g_t.cpu().numpy()
    topk_ent = geom_np[topk_local]
    recalls = []
    for i, cid in enumerate(cands):
        gt = gt_sets[i]
        if not gt: continue
        pred = [int(x) for x in topk_ent[i] if int(x) != cid][:topk]
        hits = sum(1 for p in pred if p in gt)
        recalls.append(hits / min(len(gt), topk))
    return float(np.mean(recalls)) if recalls else 0.0


def diag_scale_clustering(emb, admin_int, n=120, topk=50, device="cuda"):
    mask = admin_int >= 0
    idx = np.where(mask)[0]
    if len(idx) < n: return 0.0
    rng = np.random.default_rng(1); cands = rng.choice(idx, size=n, replace=False)
    e = emb.to(device).float()
    scores = []
    for cid in cands:
        q = e[int(cid)].unsqueeze(0)
        d = torch.norm(e - q, dim=-1)
        d[int(cid)] = float("inf")
        _, top_i = torch.topk(d, k=topk, largest=False)
        same = (admin_int[top_i.cpu().numpy()] == admin_int[int(cid)]).sum()
        scores.append(same / topk)
    return float(np.mean(scores))


def diag_topology_cos(emb, parent_map_iri, iri_arr, iri2pos, n=400):
    pairs = list(parent_map_iri.items())
    rng = np.random.default_rng(2); rng.shuffle(pairs); pairs = pairs[:n]
    cosvals = []
    for child_iri, parent_iri in pairs:
        ci = iri2pos.get(child_iri); pi = iri2pos.get(parent_iri)
        if ci is None or pi is None: continue
        ec = emb[ci]; ep = emb[pi]
        cos = (ec * ep).sum() / (ec.norm() * ep.norm() + 1e-8)
        cosvals.append(float(cos))
    return float(np.mean(cosvals)) if cosvals else 0.0


def diag_relpos_analogy(emb, has_relpos, relpos_dlat, relpos_dlon, par_idx, n=80, device="cuda"):
    idx = np.where(has_relpos > 0)[0]
    if len(idx) < n*2: return 0.0
    rng = np.random.default_rng(3); pick = rng.choice(idx, size=n*2, replace=False)
    a = pick[:n]; b = pick[n:]
    e = emb.to(device).float()
    matches = 0; valid = 0
    for ia, ib in zip(a, b):
        if par_idx[ia] == par_idx[ib]: continue
        valid += 1
        rp_dist = math.sqrt((relpos_dlat[ia]-relpos_dlat[ib])**2 + (relpos_dlon[ia]-relpos_dlon[ib])**2)
        target = math.exp(-rp_dist * 5.0)
        cos = float(F.cosine_similarity(e[int(ia)].unsqueeze(0), e[int(ib)].unsqueeze(0), -1).item())
        if (target > 0.5 and cos > 0.5) or (target <= 0.5 and cos <= 0.5):
            matches += 1
    return float(matches / valid) if valid else 0.0


def diag_area_clustering(emb, log_area_ratio, has_relpos, n=80, topk=30, device="cuda"):
    idx = np.where(has_relpos > 0)[0]
    if len(idx) < n: return 0.0
    rng = np.random.default_rng(4); cands = rng.choice(idx, size=n, replace=False)
    e = emb.to(device).float()
    scores = []
    for cid in cands:
        q = e[int(cid)].unsqueeze(0)
        d = torch.norm(e - q, dim=-1)
        d[int(cid)] = float("inf")
        _, top_i = torch.topk(d, k=topk, largest=False)
        target = log_area_ratio[int(cid)]
        peer = log_area_ratio[top_i.cpu().numpy()]
        match = (np.abs(peer - target) < 0.5).sum()
        scores.append(match / topk)
    return float(np.mean(scores))


def run_diags(emb_cpu, aux, par_idx, log_area_ratio, has_relpos, relpos_dlat, relpos_dlon,
              parent_map_iri, iri_arr, iri2pos, device="cuda"):
    d1 = diag_cross_admin_recall(emb_cpu, aux["lat"], aux["lon"], aux["has_geom"], par_idx, n=120, device=device)
    d2 = diag_scale_clustering(emb_cpu, aux["admin_int"], n=120, topk=50, device=device)
    d3 = diag_topology_cos(emb_cpu, parent_map_iri, iri_arr, iri2pos, n=400)
    d4 = diag_relpos_analogy(emb_cpu, has_relpos, relpos_dlat, relpos_dlon, par_idx, n=80, device=device)
    d5 = diag_area_clustering(emb_cpu, log_area_ratio, has_relpos, n=80, topk=30, device=device)
    composite = (d1 / 0.4 + d2 / 0.8 + d3 / 0.5 + d4 / 0.6 + d5 / 0.4) / 5
    return {"diag1": d1, "diag2": d2, "diag3": d3, "diag4": d4, "diag5": d5, "composite": composite}


# ----------- main -----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--lambda_prox", type=float, default=1.0)
    ap.add_argument("--lambda_scale", type=float, default=0.2)
    ap.add_argument("--lambda_topo", type=float, default=0.3)
    ap.add_argument("--lambda_relpos", type=float, default=0.5)
    ap.add_argument("--lambda_type", type=float, default=0.2)
    ap.add_argument("--lambda_density", type=float, default=0.1)
    ap.add_argument("--aux_batch", type=int, default=512)
    ap.add_argument("--diag_every", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--entity-id-path", required=True)
    ap.add_argument("--relation-id-path", required=True)
    ap.add_argument("--entities-path", required=True)
    ap.add_argument("--raw-features-path", required=True)
    ap.add_argument("--triples-train-path", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"config: {vars(args)}", flush=True)

    print("loading entity indexes, features, and source tables ...", flush=True)
    e2i = json.load(open(args.entity_id_path))
    r2i = json.load(open(args.relation_id_path))
    n_ent = len(e2i); n_rel = len(r2i)
    raw_features_np = np.load(args.raw_features_path)
    print(f"  n_ent={n_ent:,} n_rel={n_rel:,} raw_features={raw_features_np.shape}", flush=True)
    assert raw_features_np.shape[0] == n_ent and raw_features_np.shape[1] == 38
    df_ent = pd.read_parquet(args.entities_path)
    print(f"  entities parquet: {len(df_ent):,}", flush=True)

    aux = build_aux_features(e2i, df_ent)

    # Normalize lon/lat for encoder
    lon_norm = np.where(aux["has_geom"] > 0, (aux["lon"] - 105.0) / 35.0, 0.0).astype(np.float32)
    lat_norm = np.where(aux["has_geom"] > 0, (aux["lat"] - 35.0) / 25.0, 0.0).astype(np.float32)

    # Build geohash vocab
    def build_id_map(arr):
        uniq = [v for v in np.unique(arr) if v != ""]
        m = {v: i + 1 for i, v in enumerate(uniq)}
        ids = np.array([m.get(v, 0) for v in arr], dtype=np.int64)
        return m, ids
    gh4_map, gh4_idx = build_id_map(aux["gh4_str"])
    gh5_map, gh5_idx = build_id_map(aux["gh5_str"])
    gh6_map, gh6_idx = build_id_map(aux["gh6_str"])
    print(f"  geohash vocab: p4={len(gh4_map)} p5={len(gh5_map)} p6={len(gh6_map)}", flush=True)

    admin_raw = aux["admin_int"]
    admin_idx = np.where(admin_raw >= 0, admin_raw - 1, 0).astype(np.int64)
    admin_vocab = int(admin_idx.max()) + 2
    print(f"  admin_vocab={admin_vocab}", flush=True)

    osm_type_to_idx = {"": 0, "admin": 1, "poi": 2, "natural": 3, "transport": 4}
    osm_idx = np.array([osm_type_to_idx.get(s, 0) for s in aux["osm_type_str"]], dtype=np.int64)
    print(f"  osm_type counts: " +
          ", ".join(f"{k}={int((osm_idx==v).sum())}" for k,v in osm_type_to_idx.items() if v>0), flush=True)

    # ---- Compute relpos_aux inline (par_idx, relpos_dlat/dlon, log_area_ratio) ----
    # Parent admin map from triples: child_iri -> parent_iri (parentAdmin / parentInRegion)
    print("computing relative-position features from administrative relations ...", flush=True)
    parent_map = {}
    with open(args.triples_train_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3: continue
            h, r, t = parts
            if r in ("parentAdmin", "parentInRegion"):
                if h not in parent_map:
                    parent_map[h] = t
    print(f"  parent_map: {len(parent_map):,}", flush=True)

    iri_arr = aux["entity_iri"]
    iri2pos = {iri: i for i, iri in enumerate(iri_arr) if iri}
    par_idx = np.full(n_ent, -1, dtype=np.int64)
    for child_iri, parent_iri in parent_map.items():
        ci = iri2pos.get(child_iri); pi = iri2pos.get(parent_iri)
        if ci is not None and pi is not None:
            par_idx[ci] = pi

    has_relpos = np.zeros(n_ent, dtype=np.float32)
    relpos_dlat = np.zeros(n_ent, dtype=np.float32)
    relpos_dlon = np.zeros(n_ent, dtype=np.float32)
    log_area_ratio = np.zeros(n_ent, dtype=np.float32)
    for i in range(n_ent):
        pi = par_idx[i]
        if pi < 0: continue
        if aux["has_geom"][i] > 0 and aux["has_geom"][pi] > 0:
            has_relpos[i] = 1.0
            relpos_dlat[i] = (aux["lat"][i] - aux["lat"][pi]) / 25.0
            relpos_dlon[i] = (aux["lon"][i] - aux["lon"][pi]) / 35.0
            # log_area_ratio uses raw[2]
            log_area_ratio[i] = float(raw_features_np[i, 2] - raw_features_np[pi, 2])
    print(f"  has_relpos: {int(has_relpos.sum()):,}", flush=True)

    device = args.device
    model = GeoKGEModel(
        n_entities=n_ent, n_relations=n_rel,
        n_geohash_p4=len(gh4_map), n_geohash_p5=len(gh5_map), n_geohash_p6=len(gh6_map),
        admin_vocab=admin_vocab,
        osm_type_vocab=len(osm_type_to_idx),
        raw_features=torch.from_numpy(raw_features_np),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,} (trainable {n_train:,})", flush=True)
    print(f"  raw buffer requires_grad={model.raw_features.requires_grad} (should be False)", flush=True)

    feat_t = {k: torch.from_numpy(v).to(device) for k, v in {
        "lon_norm": lon_norm, "lat_norm": lat_norm,
        "has_geom": aux["has_geom"].astype(np.float32),
        "gh4_idx": gh4_idx, "gh5_idx": gh5_idx, "gh6_idx": gh6_idx,
        "admin_idx": admin_idx, "osm_idx": osm_idx,
        "relpos_dlat": relpos_dlat, "relpos_dlon": relpos_dlon,
        "log_area_ratio": log_area_ratio, "has_relpos": has_relpos,
    }.items()}

    has_geom_b = (aux["has_geom"] > 0)
    has_relpos_b = (has_relpos > 0)
    geom_ids = torch.nonzero(torch.from_numpy(has_geom_b).to(device), as_tuple=True)[0]
    admin_ids = torch.nonzero(torch.from_numpy(admin_raw >= 0).to(device), as_tuple=True)[0]
    relpos_ids = torch.nonzero(torch.from_numpy(has_relpos_b).to(device), as_tuple=True)[0]

    # osm_type ids per type (for L_type)
    type_ids = {}
    for t_name, t_v in osm_type_to_idx.items():
        if t_v == 0: continue
        ids = np.where(osm_idx == t_v)[0]
        if len(ids) > 0:
            type_ids[t_v] = torch.from_numpy(ids).to(device)
    print(f"  geom={len(geom_ids):,}  admin={len(admin_ids):,}  relpos={len(relpos_ids):,}  "
          f"types={ {k:len(v) for k,v in type_ids.items()} }", flush=True)

    # POI density (vec[21]) for L_density
    poi_density = torch.from_numpy(raw_features_np[:, 21].astype(np.float32)).to(device)

    def encode_idx(idx):
        return model.encode_entity(
            idx,
            feat_t["lon_norm"][idx], feat_t["lat_norm"][idx], feat_t["has_geom"][idx],
            feat_t["gh4_idx"][idx], feat_t["gh5_idx"][idx], feat_t["gh6_idx"][idx],
            feat_t["admin_idx"][idx], feat_t["osm_idx"][idx],
            feat_t["relpos_dlat"][idx], feat_t["relpos_dlon"][idx],
            feat_t["log_area_ratio"][idx], feat_t["has_relpos"][idx],
        )

    train_arr = load_triple_ids(args.triples_train_path, e2i, r2i)
    ds = TripleDS(train_arr)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                         num_workers=args.num_workers, collate_fn=collate,
                         pin_memory=True, drop_last=True)

    sf_relations = []
    for rel_name, rid in r2i.items():
        low = rel_name.lower()
        if "sfwithin" in low or "sfcoveredby" in low or "parentadmin" == low or "parentinregion" == low:
            sf_relations.append(rid)
        if "sfcontains" in low or "sfcovers" in low or "childadmin" == low:
            sf_relations.append(rid)
    sf_relations = torch.tensor(sf_relations, dtype=torch.long, device=device)
    print(f"  sf_relations: {len(sf_relations)}", flush=True)

    optim_ = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim_, start_factor=1.0, end_factor=0.1, total_iters=args.epochs)

    parent_map_iri = {child: parent for child, parent in parent_map.items()}

    log_path = out / "train_log.jsonl"
    log_f = open(log_path, "a")
    best_composite = -1.0
    best_epoch = -1

    lat_t_dev = torch.from_numpy(aux["lat"]).to(device)
    lon_t_dev = torch.from_numpy(aux["lon"]).to(device)
    admin_t_dev = torch.from_numpy(admin_raw).to(device)

    print(f"\nstarting training for {args.epochs} epochs x {len(loader)} iterations", flush=True)
    t_start = time.time()
    step = 0
    for ep in range(args.epochs):
        model.train()
        ep_t = time.time()
        sums = {"main":0.0, "prox":0.0, "scale":0.0, "topo":0.0, "relpos":0.0, "type":0.0, "density":0.0}
        n_iter = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            h = batch[:,0]; r = batch[:,1]; t = batch[:,2]
            B = h.size(0)

            neg_t = torch.randint(0, n_ent, (B,), device=device)
            neg_h = torch.randint(0, n_ent, (B,), device=device)
            corrupt = (torch.rand(B, device=device) < 0.5)
            nh = torch.where(corrupt, neg_h, h)
            nt = torch.where(corrupt, t, neg_t)

            all_idx = torch.cat([h, t, nh, nt], 0)
            all_emb = encode_idx(all_idx)
            eh, et, neh, nete = torch.chunk(all_emb, 4, 0)
            er = model.relation_emb(r)
            pos_s = model.score(eh, er, et); neg_s = model.score(neh, er, nete)
            loss_main = F.relu(args.margin - pos_s + neg_s).mean()

            # L_prox (proximity)
            if args.lambda_prox > 0 and len(geom_ids) >= args.aux_batch*3:
                pick = geom_ids[torch.randint(0, len(geom_ids), (args.aux_batch*3,), device=device)]
                a = pick[0::3]; p = pick[1::3]; q = pick[2::3]
                d_ap = haversine(lat_t_dev[a], lon_t_dev[a], lat_t_dev[p], lon_t_dev[p])
                d_aq = haversine(lat_t_dev[a], lon_t_dev[a], lat_t_dev[q], lon_t_dev[q])
                swap = d_ap > d_aq
                p2 = torch.where(swap, q, p); q2 = torch.where(swap, p, q)
                e_aux = encode_idx(torch.cat([a, p2, q2], 0))
                ea, ep_, eq = torch.chunk(e_aux, 3, 0)
                d_emb_p = torch.norm(ea - ep_, dim=-1); d_emb_q = torch.norm(ea - eq, dim=-1)
                loss_prox = F.relu(d_emb_p - d_emb_q + 1.0).mean()
            else:
                loss_prox = torch.tensor(0.0, device=device)

            # L_scale (admin clustering)
            if args.lambda_scale > 0 and len(admin_ids) >= args.aux_batch*2:
                pick = admin_ids[torch.randint(0, len(admin_ids), (args.aux_batch*2,), device=device)]
                ia = pick[0::2]; ib = pick[1::2]
                e_a = encode_idx(ia); e_b = encode_idx(ib)
                cos = F.cosine_similarity(e_a, e_b, -1)
                same = (admin_t_dev[ia] == admin_t_dev[ib]).float()
                loss_scale = F.mse_loss(cos, same)
            else:
                loss_scale = torch.tensor(0.0, device=device)

            # L_topo (sf relations)
            if args.lambda_topo > 0 and len(sf_relations) > 0:
                mask = torch.isin(r, sf_relations)
                if mask.sum() > 0:
                    sel = mask.nonzero(as_tuple=True)[0]
                    d = torch.norm(eh[sel] - et[sel], dim=-1)
                    loss_topo = F.relu(d - 2.0).mean()
                else:
                    loss_topo = torch.tensor(0.0, device=device)
            else:
                loss_topo = torch.tensor(0.0, device=device)

            # L_relpos
            if args.lambda_relpos > 0 and len(relpos_ids) >= args.aux_batch*2:
                pick = relpos_ids[torch.randint(0, len(relpos_ids), (args.aux_batch*2,), device=device)]
                ia = pick[0::2]; ib = pick[1::2]
                par_a = torch.from_numpy(par_idx).to(device)[ia]
                par_b = torch.from_numpy(par_idx).to(device)[ib]
                diff_par = par_a != par_b
                if diff_par.sum() > 16:
                    ia = ia[diff_par]; ib = ib[diff_par]
                    e_a = encode_idx(ia); e_b = encode_idx(ib)
                    rp_a = torch.stack([feat_t["relpos_dlat"][ia], feat_t["relpos_dlon"][ia]], -1)
                    rp_b = torch.stack([feat_t["relpos_dlat"][ib], feat_t["relpos_dlon"][ib]], -1)
                    rp_dist = torch.norm(rp_a - rp_b, dim=-1)
                    target = torch.exp(-rp_dist * 5.0)
                    cos = F.cosine_similarity(e_a, e_b, -1)
                    loss_relpos = F.mse_loss(cos, target)
                else:
                    loss_relpos = torch.tensor(0.0, device=device)
            else:
                loss_relpos = torch.tensor(0.0, device=device)

            # L_type (NEW: same osm_type → cos == 1, different → cos == 0)
            if args.lambda_type > 0 and len(type_ids) >= 2:
                B_t = args.aux_batch
                osm_t_dev = feat_t["osm_idx"]
                # random anchors with known type
                typed = (osm_t_dev > 0).nonzero(as_tuple=True)[0]
                if len(typed) >= B_t * 2:
                    pick = typed[torch.randint(0, len(typed), (B_t*2,), device=device)]
                    ia = pick[0::2]; ib = pick[1::2]
                    e_a = encode_idx(ia); e_b = encode_idx(ib)
                    cos = F.cosine_similarity(e_a, e_b, -1)
                    same = (osm_t_dev[ia] == osm_t_dev[ib]).float()
                    loss_type = F.mse_loss(cos, same)
                else:
                    loss_type = torch.tensor(0.0, device=device)
            else:
                loss_type = torch.tensor(0.0, device=device)

            # L_density (NEW: similar POI density → high cos)
            if args.lambda_density > 0 and len(geom_ids) >= args.aux_batch*2:
                pick = geom_ids[torch.randint(0, len(geom_ids), (args.aux_batch*2,), device=device)]
                ia = pick[0::2]; ib = pick[1::2]
                e_a = encode_idx(ia); e_b = encode_idx(ib)
                cos = F.cosine_similarity(e_a, e_b, -1)
                d_diff = (poi_density[ia] - poi_density[ib]).abs()
                target = torch.exp(-d_diff * 4.0)
                loss_density = F.mse_loss(cos, target)
            else:
                loss_density = torch.tensor(0.0, device=device)

            loss = (loss_main
                    + args.lambda_prox * loss_prox
                    + args.lambda_scale * loss_scale
                    + args.lambda_topo * loss_topo
                    + args.lambda_relpos * loss_relpos
                    + args.lambda_type * loss_type
                    + args.lambda_density * loss_density)
            optim_.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()

            sums["main"] += loss_main.item(); sums["prox"] += loss_prox.item()
            sums["scale"] += loss_scale.item(); sums["topo"] += loss_topo.item()
            sums["relpos"] += loss_relpos.item(); sums["type"] += loss_type.item()
            sums["density"] += loss_density.item()
            n_iter += 1; step += 1
            if step % 500 == 0:
                line = {"epoch": ep, "step": step,
                        **{f"loss_{k}": v/n_iter for k, v in sums.items()},
                        "loss_total": loss.item(), "elapsed": round(time.time()-t_start, 1)}
                print(f"  step {step:7d} main={loss_main.item():.3f} prox={loss_prox.item():.3f} "
                      f"scale={loss_scale.item():.4f} topo={loss_topo.item():.3f} "
                      f"relpos={loss_relpos.item():.4f} type={loss_type.item():.4f} "
                      f"dens={loss_density.item():.4f}  ({line['elapsed']:.0f}s)", flush=True)
                log_f.write(json.dumps(line)+"\n"); log_f.flush()

        ep_secs = time.time() - ep_t
        ep_avg = {k: v/max(n_iter,1) for k, v in sums.items()}
        print(f"[epoch {ep:3d}] main={ep_avg['main']:.3f} prox={ep_avg['prox']:.3f} "
              f"scale={ep_avg['scale']:.4f} topo={ep_avg['topo']:.3f} "
              f"relpos={ep_avg['relpos']:.4f} type={ep_avg['type']:.4f} "
              f"dens={ep_avg['density']:.4f}  ({ep_secs:.0f}s)", flush=True)
        scheduler.step()
        log_f.write(json.dumps({"epoch_summary": True, "epoch": ep, **ep_avg, "secs": ep_secs}) + "\n")
        log_f.flush()

        if (ep + 1) % args.diag_every == 0 or ep == args.epochs - 1:
            print(f"  [diag] epoch {ep} ...", flush=True)
            t_d = time.time()
            model.eval()
            embs = torch.zeros(n_ent, model.emb_dim, dtype=torch.float32)
            bs = 8192
            with torch.no_grad():
                for s in range(0, n_ent, bs):
                    ids = torch.arange(s, min(s+bs, n_ent), device=device)
                    embs[s:s+ids.size(0)] = encode_idx(ids).float().cpu()
            diags = run_diags(embs, aux, par_idx, log_area_ratio, has_relpos,
                              relpos_dlat, relpos_dlon, parent_map_iri,
                              iri_arr, iri2pos, device=device)
            diags["epoch"] = ep; diags["secs"] = time.time()-t_d
            print(f"    d1={diags['diag1']:.3f} d2={diags['diag2']:.3f} d3={diags['diag3']:.3f} "
                  f"d4={diags['diag4']:.3f} d5={diags['diag5']:.3f}  comp={diags['composite']:.3f} "
                  f"({diags['secs']:.1f}s)", flush=True)
            log_f.write(json.dumps({"diag_eval": True, **diags}) + "\n"); log_f.flush()
            if diags["composite"] > best_composite:
                best_composite = diags["composite"]; best_epoch = ep
                torch.save({"entity_emb": embs, "entity_to_id": e2i,
                            "embedding_dim": model.emb_dim,
                            "raw_feat_dim": model.raw_feat_dim,
                            "learned_dim": model.learned_dim,
                            "diag": diags, "epoch": ep},
                           out / "best_entity_embeddings.pt")
                torch.save({"state_dict": model.state_dict(), "args": vars(args),
                            "epoch": ep, "diag": diags},
                           out / "best_encoder.pt")
                print(f"    ★ new best (composite={best_composite:.3f}); saved", flush=True)

    print("extracting final entity embeddings ...", flush=True)
    model.eval()
    embs = torch.zeros(n_ent, model.emb_dim, dtype=torch.float32)
    bs = 8192
    with torch.no_grad():
        for s in range(0, n_ent, bs):
            ids = torch.arange(s, min(s+bs, n_ent), device=device)
            embs[s:s+ids.size(0)] = encode_idx(ids).float().cpu()
    torch.save({"entity_emb": embs, "entity_to_id": e2i, "embedding_dim": model.emb_dim,
                "raw_feat_dim": model.raw_feat_dim, "learned_dim": model.learned_dim},
               out / "entity_embeddings.pt")

    raw_check = model.raw_features.cpu().float()
    diff = (embs[:, :38] - raw_check).abs().max().item()
    print(f"feature alignment: max |emb[:38] - raw_features| = {diff:.6e}", flush=True)

    with open(out / "feature_index.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch, "best_composite": best_composite,
            "raw_feat_dim": 38, "learned_dim": 218, "emb_dim": 256,
            "raw_features_path": args.raw_features_path,
            "n_entities": n_ent, "n_relations": n_rel,
        }, f, indent=2)
    log_f.close()
    print(f"DONE  best epoch={best_epoch} composite={best_composite:.3f}  "
          f"total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
