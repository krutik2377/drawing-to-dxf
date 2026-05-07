"""Skeleton pixel graph → traced polylines (networkx, 8-connectivity)."""

from __future__ import annotations

import networkx as nx
import numpy as np

Offset = tuple[int, int]

_NEIGH8: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def skeleton_adjacency_graph(ink: np.ndarray) -> nx.Graph:
    """8-connected skeleton graph from boolean/binary mask (nonzero = vertex)."""
    g = nx.Graph()
    ys, xs = np.nonzero(ink)
    for r, c in zip(ys, xs, strict=True):
        r_i, c_i = int(r), int(c)
        p = (r_i, c_i)
        g.add_node(p)
        for dr, dc in _NEIGH8:
            rr, cc = r_i + dr, c_i + dc
            if rr < 0 or cc < 0 or rr >= ink.shape[0] or cc >= ink.shape[1]:
                continue
            if ink[rr, cc]:
                q = (rr, cc)
                g.add_edge(p, q)
    return g


def _adjacent_skeleton_nodes(gray: np.ndarray) -> nx.Graph:
    """Build graph from thresholded uint8 skeleton (255 = foreground)."""
    fg = gray > 0
    return skeleton_adjacency_graph(fg)


def _walk_chain(
    hsn: nx.Graph,
    *,
    anchor: Offset,
    nxt: Offset,
    specials: set[Offset],
) -> tuple[list[Offset], set[frozenset[Offset]]]:
    """Follow unique degree-2 path from anchor→nxt; stop at specials or branching."""
    path: list[Offset] = [anchor, nxt]
    used: set[frozenset[Offset]] = {frozenset((anchor, nxt))}
    prev, curr = anchor, nxt
    guard = max(hsn.number_of_nodes(), 64) + 8
    for _ in range(guard):
        if curr != anchor and curr in specials:
            return path, used
        successors = [n for n in hsn.neighbors(curr) if n != prev]
        if len(successors) != 1:
            return path, used
        nxt_px = successors[0]
        used.add(frozenset((curr, nxt_px)))
        prev, curr = curr, nxt_px
        path.append(curr)
        if curr == anchor and len(path) > 3:
            return path, used
    return path, used


def trace_skeleton_polylines(ink_skeleton: np.ndarray) -> tuple[list[list[tuple[float, float]]], list[bool]]:
    """
    Skeleton boolean / uint8 mask (nonzero strokes).

    Returns polylines in image XY (column, row) floats, with closed flags for loops.
    """
    if ink_skeleton.ndim != 2:
        raise ValueError("2D skeleton expected")

    fg = ink_skeleton.astype(bool)
    ink_u8 = (fg.astype(np.uint8)) * np.uint8(255)
    adj = _adjacent_skeleton_nodes(ink_u8)
    if adj.number_of_nodes() == 0:
        return [], []

    polylines_px: list[list[tuple[float, float]]] = []
    closed_flags: list[bool] = []
    used_edges_global: set[frozenset[Offset]] = set()

    def to_xy(rrc: Offset) -> tuple[float, float]:
        return (float(rrc[1]), float(rrc[0]))

    def record_open(path_px: list[Offset]) -> None:
        if len(path_px) < 2:
            return
        xy = [to_xy(p) for p in path_px]
        if len({(round(x, 3), round(y, 3)) for x, y in xy}) < 2:
            return
        polylines_px.append(xy)
        closed_flags.append(False)

    def record_closed(path_px: list[Offset]) -> None:
        dedup_tail = path_px[:-1] if len(path_px) >= 4 and path_px[0] == path_px[-1] else path_px
        xy = [to_xy(p) for p in dedup_tail]
        if len(xy) < 3:
            return
        polylines_px.append(xy)
        closed_flags.append(True)

    def absorb_edges(us: set[frozenset[Offset]]) -> None:
        used_edges_global.update(us)

    for comp in nx.connected_components(adj):
        hsn = adj.subgraph(comp).copy()

        specials: set[Offset] = set()
        for n in hsn.nodes:
            d = int(hsn.degree[n])
            if d != 2:
                specials.add(n)

        # Pure cyclic chain (degree 2 everywhere)
        if not specials:
            anchor = next(iter(hsn.nodes))
            neigh = next(iter(hsn.neighbors(anchor)))
            chain, ue = _walk_chain(hsn, anchor=anchor, nxt=neigh, specials=set())
            absorb_edges(ue)
            if chain[0] == chain[-1] and len(chain) >= 5:
                record_closed(chain + [])
            elif len(chain) >= 2:
                record_open(chain)
            continue

        # Start from specials to explore incident edges once
        for s in specials:
            for nb in list(hsn.neighbors(s)):
                fe = frozenset((s, nb))
                if fe in used_edges_global:
                    continue
                chain, ue = _walk_chain(hsn, anchor=s, nxt=nb, specials=specials)
                absorb_edges(ue)

                closed_loop = chain[0] == chain[-1] and len(chain) >= 4
                hits_other_special_interior = closed_loop and chain[-2] != chain[1]

                if closed_loop and hits_other_special_interior:
                    record_closed(chain)
                elif closed_loop:
                    record_closed(chain)
                else:
                    record_open(chain)

        # Loose edges left (isolated bends / tiny branches)
        for u, v in hsn.edges:
            fe = frozenset((u, v))
            if fe not in used_edges_global:
                chain, ue = _walk_chain(hsn, anchor=u, nxt=v, specials=specials)
                absorb_edges(ue)
                if chain[0] == chain[-1] and len(chain) >= 4:
                    record_closed(chain)
                else:
                    record_open(chain)

    return polylines_px, closed_flags
