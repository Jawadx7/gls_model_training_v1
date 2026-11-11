"""
Static reference: GhanaLandScan tile name -> database tileId.
Used by build_tile_metadata.py to attach human-readable names to the raw export.

Deliberately excludes AS-(Kokofu) and AS-(Nyinahin)-1 — not part of the agreed 24-tile set.
"""

NAME_TO_ID = {
    "AS-(Adubia)": "cmqmwak2y0001l504espuqmxm",
    "AS-(Anwia-Nkwanta)": "cmqz8r74z0001l2041pv6yyca",
    "AS-(Anyinam)": "cmqzd6zxa0001jt04drxnzhgu",
    "AS-(Asankrangwa)": "cmqz5r6pj0001i90430ije0vt",
    "AS-(Bibiani)": "cmr6gsbkm0001l804k51nf4l5",
    "AS-(Dokyiwa)": "cmqzaawz50001jr042o9q76en",
    "AS-(Esaase)": "cmqmet3mu0001k004zpljf6ho",
    "AS-(Fomena)": "cmqv3fcih0001jx04pcv6eqkn",
    "AS-(Fomena)-1": "cmr6j8hyy0003jt04rb418g1p",
    "AS-(Jacobu)": "cmqmg5w6y0005k004w6l56ryb",
    "AS-(kunsu)": "cmqzdvst70003jt04cdua23j6",
    "AS-(Manso-Adubia)": "cmqv3t8bp0003jx041ml95i5k",
    "AS-(Nkroful)": "cmqz743010001js0436yxiba1",
    "AS-(Nyinahin)": "cmr6fiw290001ib04uxs3da1c",
    "AS-(Poano)": "cmr6ito8i0001jt04i2lq143z",
    "AS-(Samreboi)": "cmqz62vnb0003i904ws5hjow5",
    "AS-(Sanso)": "cmqz8yafz0003l204x1lat6p0",
    "AS-(Tweapease)": "cmqmhxymt0001js046ck4sli6",
    "AS-001": "cmppmb8lh0001jp04ls2o04q6",
    "AS-002": "cmppn7tng0003jp04q5yshgew",
    "AS-003": "cmpps3ptq0003l504p0p2pit4",
    "AS-005": "cmpps7p2f0007l504hke00m74",
    "AS-006": "cmppsazv10009l504smotc20o",
    "AS-007": "cmppsea4d000bl504t36zndw7",
}

ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
