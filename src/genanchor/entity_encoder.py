"""Spatial entity encoder used to build the GeoAnchor entity table.

Architecture:

  entity_emb(i) = concat(
      raw_features[i, :38],     # 38 dims, frozen, hard-coded (lon/lat/area/admin_one_hot/...
                                #         /poi_density/road_density/river/coast/elev/...
                                #         /osm_type_one_hot/population/year_built/.../geohash)
      encoder_output[i, :218],  # 218 dims, learned via TransE + auxiliary losses
  )                             # = 256 dims total

The encoder combines entity identifiers, coordinates, multi-precision geohash,
administrative context, relative position, and object type information.

Decoupling guarantee: entity_emb[:, :38] == raw_features[:, :38] exactly. The TransE
objective operates on the full 256-d embedding so raw features participate as INPUT
to relational reasoning (not as a regression target).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class GeoKGEModel(nn.Module):
    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        n_geohash_p4: int,
        n_geohash_p5: int,
        n_geohash_p6: int,
        raw_features: torch.Tensor,       # (n_entities, 38) float32
        emb_dim: int = 256,
        raw_feat_dim: int = 38,
        id_dim: int = 128,
        lonlat_dim: int = 64,
        geohash_dim: int = 32,
        admin_dim: int = 16,
        relpos_dim: int = 32,
        osm_type_dim: int = 16,
        admin_vocab: int = 12,
        osm_type_vocab: int = 5,  # 0=none, 1=admin, 2=poi, 3=natural, 4=transport
    ):
        super().__init__()
        assert raw_features.shape == (n_entities, raw_feat_dim), \
            f"raw_features shape {tuple(raw_features.shape)} != ({n_entities}, {raw_feat_dim})"
        self.n_entities = n_entities
        self.emb_dim = emb_dim
        self.raw_feat_dim = raw_feat_dim
        self.learned_dim = emb_dim - raw_feat_dim
        assert self.learned_dim > 0

        self.register_buffer("raw_features", raw_features.float())

        self.id_emb    = nn.Embedding(n_entities, id_dim)
        self.gh4_emb   = nn.Embedding(n_geohash_p4 + 1, geohash_dim, padding_idx=0)
        self.gh5_emb   = nn.Embedding(n_geohash_p5 + 1, geohash_dim, padding_idx=0)
        self.gh6_emb   = nn.Embedding(n_geohash_p6 + 1, geohash_dim, padding_idx=0)
        self.admin_emb = nn.Embedding(admin_vocab, admin_dim, padding_idx=0)
        self.osm_emb   = nn.Embedding(osm_type_vocab, osm_type_dim, padding_idx=0)
        self.lonlat_mlp = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, lonlat_dim),
        )
        self.relpos_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, relpos_dim),
        )
        concat_dim = id_dim + lonlat_dim + 3 * geohash_dim + admin_dim + osm_type_dim + relpos_dim
        self.proj = nn.Linear(concat_dim, self.learned_dim)

        self.relation_emb = nn.Embedding(n_relations, emb_dim)

        nn.init.xavier_uniform_(self.id_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        for E in (self.gh4_emb, self.gh5_emb, self.gh6_emb, self.admin_emb, self.osm_emb):
            nn.init.xavier_uniform_(E.weight); E.weight.data[0].zero_()

    def encode_entity(
        self,
        idx,
        lon, lat, has_geom,
        gh4, gh5, gh6,
        admin_idx,
        osm_idx,
        relpos_dlat, relpos_dlon, log_area_ratio, has_relpos,
    ):
        id_v = self.id_emb(idx)
        ll = torch.stack([lon, lat], dim=-1).to(id_v.dtype)
        ll_v = self.lonlat_mlp(ll) * has_geom.to(id_v.dtype).unsqueeze(-1)
        rp = torch.stack([relpos_dlat, relpos_dlon, log_area_ratio], dim=-1).to(id_v.dtype)
        rp_v = self.relpos_mlp(rp) * has_relpos.to(id_v.dtype).unsqueeze(-1)
        gh4_v = self.gh4_emb(gh4); gh5_v = self.gh5_emb(gh5); gh6_v = self.gh6_emb(gh6)
        ad_v = self.admin_emb(admin_idx)
        os_v = self.osm_emb(osm_idx)
        cat = torch.cat([id_v, ll_v, gh4_v, gh5_v, gh6_v, ad_v, os_v, rp_v], dim=-1)
        learned = self.proj(cat)
        raw = self.raw_features[idx].to(learned.dtype)
        return torch.cat([raw, learned], dim=-1)

    def score(self, eh, er, et):
        return -torch.norm(eh + er - et, p=2, dim=-1)
