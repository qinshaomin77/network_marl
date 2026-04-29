# -*- coding: utf-8 -*-
"""
xml_mapping.py
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import xml.etree.ElementTree as ET

import numpy as np


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class ConnectionInfo:
    """单条受控连接信息（以 controlled connection 为节点）"""
    link_index: int
    incoming_lane: str
    outgoing_lane: str
    movement: str           # right / straight / left / unknown
    detector_id: str


@dataclass
class MappingBundle:
    """单个交叉口的静态映射数据包"""

    net_file: Path
    add_file: Path
    output_csv: Optional[Path] = None

    traffic_light_id: str = ""
    junction_id: str = ""

    # 连接信息
    connections: List[ConnectionInfo] = field(default_factory=list)

    # 节点顺序（connection-level nodes）
    ordered_incoming_lanes: List[str] = field(default_factory=list)
    ordered_detector_ids: List[str] = field(default_factory=list)
    ordered_link_indices: List[int] = field(default_factory=list)

    # 映射字典
    lane_to_detector: Dict[str, str] = field(default_factory=dict)
    detector_to_lane: Dict[str, str] = field(default_factory=dict)
    lane_to_link_index: Dict[str, int] = field(default_factory=dict)
    link_index_to_lane: Dict[int, str] = field(default_factory=dict)

    # 相位信息
    rl_phases: List[str] = field(default_factory=list)
    action_to_phase: Dict[int, int] = field(default_factory=dict)
    num_actions: int = 0
    all_phases: List[str] = field(default_factory=list)
    phase_durations: List[int] = field(default_factory=list)

    # 右转
    right_turn_indices: Set[int] = field(default_factory=set)

    # GraphEncoder 静态图信息
    movement_list: List[str] = field(default_factory=list)
    movement_onehot: Optional[np.ndarray] = None
    right_turn_mask: Optional[np.ndarray] = None
    self_mask: Optional[np.ndarray] = None
    same_group_edges: List[Tuple[int, int]] = field(default_factory=list)
    diff_group_edges: List[Tuple[int, int]] = field(default_factory=list)
    same_group_mask: Optional[np.ndarray] = None
    diff_group_mask: Optional[np.ndarray] = None
    phase_green_masks: Dict[int, np.ndarray] = field(default_factory=dict)

    warnings: List[str] = field(default_factory=list)


@dataclass
class NetworkGraph:
    """
    供上层 / network_embedding.py 使用的交叉口级静态图
    """
    tls_ids: List[str]
    tls_neighbors: Dict[str, Dict[str, str]]
    adj_matrix: np.ndarray
    id_to_idx: Dict[str, int]
    idx_to_id: Dict[int, str]
    node_embeddings: Optional[np.ndarray] = None




@dataclass
class EdgeGraphBundle:
    """Static edge-level graph for upper layer."""
    edge_ids: List[str]
    edge_id_to_idx: Dict[str, int]
    idx_to_edge_id: Dict[int, str]
    edge_lanes: Dict[str, List[str]]
    edge_capacity: Dict[str, float]
    A_edge_up: np.ndarray
    A_edge_down: np.ndarray
    tls_incoming_edges: Dict[str, List[str]]
    edge_to_tls: Dict[str, str]
    edge_to_tls_matrix: np.ndarray

@dataclass
class NetworkMappingBundle:
    """
    路网级静态映射总包
    """
    net_file: Path
    add_file: Path
    mapping_dict: Dict[str, MappingBundle] = field(default_factory=dict)
    network_graph: Optional[NetworkGraph] = None
    edge_graph: Optional[EdgeGraphBundle] = None
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# 基础工具函数
# =============================================================================

def _read_xml_root(xml_path: Path) -> ET.Element:
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    return ET.parse(xml_path).getroot()


def _movement_from_dir(dir_code: str) -> str:
    code = (dir_code or "").lower().strip()
    mapping = {
        "r": "right",
        "s": "straight",
        "l": "left",
        "t": "turn",
    }
    return mapping.get(code, code or "unknown")


def _movement_onehot(movement: str) -> np.ndarray:
    """
    movement -> onehot [right, straight, left]
    """
    vec = np.zeros(3, dtype=np.float32)
    if movement == "right":
        vec[0] = 1.0
    elif movement == "straight":
        vec[1] = 1.0
    elif movement == "left":
        vec[2] = 1.0
    return vec


def _safe_float(x: Optional[str], default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# =============================================================================
# 单路口静态解析
# =============================================================================

def _find_all_traffic_lights(net_root: ET.Element) -> List[str]:
    """返回 net.xml 中全部 tlLogic id"""
    tl_ids = [tl.get("id") for tl in net_root.findall("tlLogic") if tl.get("id")]
    tl_ids = sorted(set(tl_ids))
    if not tl_ids:
        raise ValueError("No traffic light (tlLogic) found in net.xml")
    return tl_ids


def _find_junction(net_root: ET.Element, tl_id: str) -> ET.Element:
    """查找 traffic_light junction"""
    junctions = net_root.findall("junction[@type='traffic_light']")
    if not junctions:
        raise ValueError("No traffic_light junction found in net.xml")

    for junction in junctions:
        if junction.get("id", "") == tl_id:
            return junction

    raise ValueError(f"Cannot find traffic_light junction for tl_id='{tl_id}'")


def _parse_controlled_connections(
    net_root: ET.Element,
    tl_id: str,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """
    解析受该 tl 控制的 controlled connections，按 linkIndex 排序
    """
    connections: List[Dict[str, Any]] = []

    for conn in net_root.findall("connection"):
        if conn.get("tl") != tl_id:
            continue

        try:
            link_index = int(conn.get("linkIndex", "-1"))
            if link_index < 0:
                continue

            from_edge = conn.get("from", "")
            from_lane_attr = conn.get("fromLane", "0")
            to_edge = conn.get("to", "")
            to_lane_attr = conn.get("toLane", "0")
            dir_code = conn.get("dir", "")

            incoming_lane = f"{from_edge}_{from_lane_attr}"
            outgoing_lane = f"{to_edge}_{to_lane_attr}"

            connections.append({
                "link_index": link_index,
                "incoming_lane": incoming_lane,
                "outgoing_lane": outgoing_lane,
                "movement": _movement_from_dir(dir_code),
            })

        except (ValueError, AttributeError) as e:
            warnings.append(f"[{tl_id}] Error parsing connection: {e}")

    if not connections:
        raise ValueError(f"No controlled connections found for traffic light '{tl_id}'")

    connections.sort(key=lambda x: x["link_index"])

    # linkIndex 连续性检查
    link_indices = [c["link_index"] for c in connections]
    expected = list(range(len(link_indices)))
    if link_indices != expected:
        warnings.append(
            f"[{tl_id}] Controlled connections linkIndex not continuous: "
            f"got {link_indices}, expected {expected}"
        )

    return connections


def _parse_detectors(add_root: ET.Element) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    解析 laneAreaDetector
    返回：
        lane_to_detector
        detector_to_lane
    """
    lane_to_detector: Dict[str, str] = {}
    detector_to_lane: Dict[str, str] = {}

    for detector in add_root.findall("laneAreaDetector"):
        det_id = detector.get("id", "")
        lane_id = detector.get("lane", "")
        if not det_id or not lane_id:
            continue

        detector_to_lane[det_id] = lane_id

        # 同一 lane 多 detector 时，优先使用 lane 同名 detector
        if lane_id not in lane_to_detector:
            lane_to_detector[lane_id] = det_id
        elif det_id == lane_id:
            lane_to_detector[lane_id] = det_id

    return lane_to_detector, detector_to_lane


def _parse_traffic_light_phases(
    net_root: ET.Element,
    tl_id: str,
    connections: List[Dict[str, Any]],
    warnings: List[str],
) -> Tuple[List[str], List[int], List[str], Dict[int, int], Set[int]]:
    """
    解析 tlLogic:
    - all_phases
    - phase_durations
    - rl_phases（排除纯黄灯，且必须存在非右转绿灯）
    - action_to_phase
    - right_turn_indices
    """
    right_turn_indices: Set[int] = set()
    for conn in connections:
        if conn["movement"] == "right":
            right_turn_indices.add(conn["link_index"])

    tl_logic = None
    for tl in net_root.findall("tlLogic"):
        if tl.get("id") == tl_id:
            tl_logic = tl
            break

    if tl_logic is None:
        raise ValueError(f"Cannot find tlLogic for traffic light '{tl_id}'")

    all_phases: List[str] = []
    phase_durations: List[int] = []

    for phase_elem in tl_logic.findall("phase"):
        state = phase_elem.get("state", "")
        duration = int(phase_elem.get("duration", "0"))
        if state:
            all_phases.append(state)
            phase_durations.append(duration)

    if not all_phases:
        raise ValueError(f"No phases found in tlLogic '{tl_id}'")

    rl_phases: List[str] = []
    action_to_phase: Dict[int, int] = {}

    for phase_idx, phase_state in enumerate(all_phases):
        if "y" in phase_state.lower():
            continue

        has_green_non_right = False
        limit = min(len(phase_state), len(connections))

        for link_idx in range(limit):
            signal = phase_state[link_idx]
            if link_idx in right_turn_indices:
                continue
            if signal.lower() == "g":
                has_green_non_right = True
                break

        if has_green_non_right:
            action_idx = len(rl_phases)
            rl_phases.append(phase_state)
            action_to_phase[action_idx] = phase_idx

    if not rl_phases:
        warnings.append(
            f"[{tl_id}] No valid green RL phases found, fallback to all phases."
        )
        rl_phases = list(all_phases)
        action_to_phase = {i: i for i in range(len(all_phases))}

    return all_phases, phase_durations, rl_phases, action_to_phase, right_turn_indices


def _build_phase_green_masks(
    rl_phases: List[str],
    num_nodes: int,
    right_turn_indices: Set[int],
) -> Dict[int, np.ndarray]:
    phase_green_masks: Dict[int, np.ndarray] = {}

    for action_idx, phase_state in enumerate(rl_phases):
        mask = np.zeros(num_nodes, dtype=np.float32)
        limit = min(len(phase_state), num_nodes)

        for i in range(limit):
            mask[i] = 1.0 if phase_state[i].lower() == "g" else 0.0

        # 右转节点：始终视作绿灯
        for i in right_turn_indices:
            if 0 <= i < num_nodes:
                mask[i] = 1.0

        phase_green_masks[action_idx] = mask

    return phase_green_masks


def _build_graph_structure(
    connections: List[Dict[str, Any]],
    rl_phases: List[str],
    right_turn_indices: Set[int],
    warnings: List[str],
) -> Dict[str, Any]:
    """
    构建单路口局部静态图结构
    """
    num_nodes = len(connections)

    movement_list = [conn["movement"] for conn in connections]
    movement_onehot = np.stack(
        [_movement_onehot(m) for m in movement_list], axis=0
    ).astype(np.float32)

    right_turn_mask = np.zeros(num_nodes, dtype=np.float32)
    for idx in right_turn_indices:
        if 0 <= idx < num_nodes:
            right_turn_mask[idx] = 1.0

    self_mask = np.eye(num_nodes, dtype=np.float32)

    # 注意：这里调用新版 _build_phase_green_masks
    phase_green_masks = _build_phase_green_masks(
        rl_phases=rl_phases,
        num_nodes=num_nodes,
        right_turn_indices=right_turn_indices,
    )

    # ---------------------------------------------------------
    # same-group
    # ---------------------------------------------------------
    same_group_mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    same_group_edges_set: Set[Tuple[int, int]] = set()

    # 1) 按每个 phase 的绿灯组构造同组边
    for green_mask in phase_green_masks.values():
        green_indices = np.where(green_mask > 0.5)[0].tolist()

        for p in range(len(green_indices)):
            for q in range(p + 1, len(green_indices)):
                i = green_indices[p]
                j = green_indices[q]
                a, b = (i, j) if i < j else (j, i)
                same_group_edges_set.add((a, b))
                same_group_mask[a, b] = 1.0
                same_group_mask[b, a] = 1.0

    # 2) 显式保证：右转节点与所有其他有效节点都属于 same-group
    #    更严格贴合“right-turn lane can be regarded as the same group as any other node”
    for i in right_turn_indices:
        if not (0 <= i < num_nodes):
            continue
        for j in range(num_nodes):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            same_group_edges_set.add((a, b))
            same_group_mask[a, b] = 1.0
            same_group_mask[b, a] = 1.0

    same_group_edges = sorted(list(same_group_edges_set))

    # ---------------------------------------------------------
    # diff-group
    # ---------------------------------------------------------
    diff_group_mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    diff_group_edges: List[Tuple[int, int]] = []

    # 只在非右转节点之间构造 diff-group
    non_right_indices = [i for i in range(num_nodes) if i not in right_turn_indices]

    for p in range(len(non_right_indices)):
        for q in range(p + 1, len(non_right_indices)):
            i = non_right_indices[p]
            j = non_right_indices[q]

            # 已经属于 same-group 的，不再进 diff-group
            if same_group_mask[i, j] > 0.5:
                continue

            diff_group_mask[i, j] = 1.0
            diff_group_mask[j, i] = 1.0
            diff_group_edges.append((i, j))

    overlap = set(same_group_edges).intersection(set(diff_group_edges))
    if overlap:
        warnings.append(
            f"same_group_edges and diff_group_edges overlap: "
            f"{sorted(list(overlap))[:10]}"
        )

    return {
        "movement_list": movement_list,
        "movement_onehot": movement_onehot,
        "right_turn_mask": right_turn_mask,
        "self_mask": self_mask,
        "same_group_edges": same_group_edges,
        "diff_group_edges": diff_group_edges,
        "same_group_mask": same_group_mask,
        "diff_group_mask": diff_group_mask,
        "phase_green_masks": phase_green_masks,
    }


def _build_bundle(
    net_file: Path,
    add_file: Path,
    output_csv: Optional[Path],
    tl_id: str,
    junction: ET.Element,
    connections: List[Dict[str, Any]],
    lane_to_detector: Dict[str, str],
    detector_to_lane: Dict[str, str],
    all_phases: List[str],
    phase_durations: List[int],
    rl_phases: List[str],
    action_to_phase: Dict[int, int],
    right_turn_indices: Set[int],
    graph_info: Dict[str, Any],
    warnings: List[str],
) -> MappingBundle:
    bundle = MappingBundle(
        net_file=net_file,
        add_file=add_file,
        output_csv=output_csv,
        traffic_light_id=tl_id,
        junction_id=junction.get("id", ""),
        warnings=list(warnings),
    )

    for conn_dict in connections:
        lane_id = conn_dict["incoming_lane"]
        detector_id = lane_to_detector.get(lane_id, lane_id)

        bundle.connections.append(
            ConnectionInfo(
                link_index=conn_dict["link_index"],
                incoming_lane=lane_id,
                outgoing_lane=conn_dict["outgoing_lane"],
                movement=conn_dict["movement"],
                detector_id=detector_id,
            )
        )

    bundle.ordered_incoming_lanes = [c.incoming_lane for c in bundle.connections]
    bundle.ordered_detector_ids = [c.detector_id for c in bundle.connections]
    bundle.ordered_link_indices = [c.link_index for c in bundle.connections]

    bundle.lane_to_detector = {c.incoming_lane: c.detector_id for c in bundle.connections}
    bundle.detector_to_lane = dict(detector_to_lane)
    bundle.lane_to_link_index = {c.incoming_lane: c.link_index for c in bundle.connections}
    bundle.link_index_to_lane = {c.link_index: c.incoming_lane for c in bundle.connections}

    bundle.all_phases = list(all_phases)
    bundle.phase_durations = list(phase_durations)
    bundle.rl_phases = list(rl_phases)
    bundle.action_to_phase = dict(action_to_phase)
    bundle.num_actions = len(rl_phases)
    bundle.right_turn_indices = set(right_turn_indices)

    bundle.movement_list = graph_info["movement_list"]
    bundle.movement_onehot = graph_info["movement_onehot"]
    bundle.right_turn_mask = graph_info["right_turn_mask"]
    bundle.self_mask = graph_info["self_mask"]
    bundle.same_group_edges = graph_info["same_group_edges"]
    bundle.diff_group_edges = graph_info["diff_group_edges"]
    bundle.same_group_mask = graph_info["same_group_mask"]
    bundle.diff_group_mask = graph_info["diff_group_mask"]
    bundle.phase_green_masks = graph_info["phase_green_masks"]

    return bundle


def _build_single_mapping_for_tl(
    net_file: Path,
    add_file: Path,
    net_root: ET.Element,
    add_root: ET.Element,
    tl_id: str,
    output_csv: Optional[Path] = None,
) -> MappingBundle:
    warnings: List[str] = []

    junction = _find_junction(net_root, tl_id)
    connections = _parse_controlled_connections(net_root, tl_id, warnings)
    lane_to_detector, detector_to_lane = _parse_detectors(add_root)

    all_phases, phase_durations, rl_phases, action_to_phase, right_turn_indices = \
        _parse_traffic_light_phases(
            net_root=net_root,
            tl_id=tl_id,
            connections=connections,
            warnings=warnings,
        )

    graph_info = _build_graph_structure(
        connections=connections,
        rl_phases=rl_phases,
        right_turn_indices=right_turn_indices,
        warnings=warnings,
    )

    return _build_bundle(
        net_file=net_file,
        add_file=add_file,
        output_csv=output_csv,
        tl_id=tl_id,
        junction=junction,
        connections=connections,
        lane_to_detector=lane_to_detector,
        detector_to_lane=detector_to_lane,
        all_phases=all_phases,
        phase_durations=phase_durations,
        rl_phases=rl_phases,
        action_to_phase=action_to_phase,
        right_turn_indices=right_turn_indices,
        graph_info=graph_info,
        warnings=warnings,
    )


# =============================================================================
# CSV 导出（可选）
# =============================================================================

def _write_csv(bundle: MappingBundle) -> None:
    if bundle.output_csv is None:
        return

    bundle.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "link_index",
        "incoming_lane",
        "outgoing_lane",
        "movement",
        "detector_id",
    ]

    rows = []
    for conn in bundle.connections:
        rows.append({
            "link_index": conn.link_index,
            "incoming_lane": conn.incoming_lane,
            "outgoing_lane": conn.outgoing_lane,
            "movement": conn.movement,
            "detector_id": conn.detector_id,
        })

    with bundle.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 路网邻接构建
# =============================================================================

def _split_grid_like_id(tls_id: str) -> Optional[Tuple[int, int]]:
    """
    从类似 00 / nt11 / tls_0304 解析 (row, col)
    规则：
    - 取末尾连续数字串
    - 长度为偶数且 >= 2
    - 前半 row，后半 col
    """
    m = re.search(r"(\d+)$", tls_id)
    if not m:
        return None

    s = m.group(1)
    if len(s) < 2 or len(s) % 2 != 0:
        return None

    half = len(s) // 2
    try:
        row = int(s[:half])
        col = int(s[half:])
        return row, col
    except Exception:
        return None


def _build_tls_neighbors_from_grid_ids(
    tls_ids: List[str],
) -> Optional[Dict[str, Dict[str, str]]]:
    """
    若所有 tls_id 都能解析成 (row, col)，则按规则网格构建邻接
    返回 None 表示不能用该方法
    """
    parsed: Dict[str, Tuple[int, int]] = {}
    for tid in tls_ids:
        rc = _split_grid_like_id(tid)
        if rc is None:
            return None
        parsed[tid] = rc

    rc_to_id = {rc: tid for tid, rc in parsed.items()}
    neighbors: Dict[str, Dict[str, str]] = {}

    for tid, (r, c) in parsed.items():
        nbrs: Dict[str, str] = {}
        if (r - 1, c) in rc_to_id:
            nbrs["n"] = rc_to_id[(r - 1, c)]
        if (r, c + 1) in rc_to_id:
            nbrs["e"] = rc_to_id[(r, c + 1)]
        if (r + 1, c) in rc_to_id:
            nbrs["s"] = rc_to_id[(r + 1, c)]
        if (r, c - 1) in rc_to_id:
            nbrs["w"] = rc_to_id[(r, c - 1)]
        neighbors[tid] = nbrs

    return neighbors


def _get_tl_junction_positions(
    net_root: ET.Element,
    tls_ids: List[str],
) -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}
    tls_set = set(tls_ids)

    for junction in net_root.findall("junction[@type='traffic_light']"):
        jid = junction.get("id", "")
        if jid not in tls_set:
            continue
        x = _safe_float(junction.get("x"), 0.0)
        y = _safe_float(junction.get("y"), 0.0)
        pos[jid] = (x, y)

    return pos


def _build_tls_neighbors_from_coordinates(
    net_root: ET.Element,
    tls_ids: List[str],
    tol: float = 1e-6,
) -> Dict[str, Dict[str, str]]:
    """
    从 junction 几何坐标构建北/东/南/西邻居
    作为非规则命名时的 fallback
    """
    pos = _get_tl_junction_positions(net_root, tls_ids)
    neighbors: Dict[str, Dict[str, str]] = {tid: {} for tid in tls_ids}

    for tid in tls_ids:
        if tid not in pos:
            continue

        x0, y0 = pos[tid]

        north_candidate = None
        south_candidate = None
        east_candidate = None
        west_candidate = None

        north_dist = float("inf")
        south_dist = float("inf")
        east_dist = float("inf")
        west_dist = float("inf")

        for other in tls_ids:
            if other == tid or other not in pos:
                continue

            x1, y1 = pos[other]

            # 同列：north / south
            if abs(x1 - x0) <= tol:
                dy = y1 - y0
                if dy > tol and dy < north_dist:
                    north_dist = dy
                    north_candidate = other
                elif dy < -tol and (-dy) < south_dist:
                    south_dist = -dy
                    south_candidate = other

            # 同行：east / west
            if abs(y1 - y0) <= tol:
                dx = x1 - x0
                if dx > tol and dx < east_dist:
                    east_dist = dx
                    east_candidate = other
                elif dx < -tol and (-dx) < west_dist:
                    west_dist = -dx
                    west_candidate = other

        if north_candidate is not None:
            neighbors[tid]["n"] = north_candidate
        if east_candidate is not None:
            neighbors[tid]["e"] = east_candidate
        if south_candidate is not None:
            neighbors[tid]["s"] = south_candidate
        if west_candidate is not None:
            neighbors[tid]["w"] = west_candidate

    return neighbors


def _build_tls_neighbors(
    net_root: ET.Element,
    tls_ids: List[str],
) -> Dict[str, Dict[str, str]]:
    """
    优先按规则网格命名构建邻接，失败后按坐标 fallback
    """
    by_grid = _build_tls_neighbors_from_grid_ids(tls_ids)
    if by_grid is not None:
        return by_grid

    return _build_tls_neighbors_from_coordinates(net_root, tls_ids)


def _build_tls_adjacency(
    tls_ids: List[str],
    tls_neighbors: Dict[str, Dict[str, str]],
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    tls_ids = list(tls_ids)
    id_to_idx = {tid: i for i, tid in enumerate(tls_ids)}
    idx_to_id = {i: tid for i, tid in enumerate(tls_ids)}

    n = len(tls_ids)
    adj = np.zeros((n, n), dtype=np.float32)

    for tid, nbrs in tls_neighbors.items():
        i = id_to_idx[tid]
        for nb_id in nbrs.values():
            if nb_id not in id_to_idx:
                continue
            j = id_to_idx[nb_id]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return adj, id_to_idx, idx_to_id


def _build_network_graph(
    net_root: ET.Element,
    tls_ids: List[str],
) -> NetworkGraph:
    tls_neighbors = _build_tls_neighbors(net_root, tls_ids)
    adj_matrix, id_to_idx, idx_to_id = _build_tls_adjacency(tls_ids, tls_neighbors)

    return NetworkGraph(
        tls_ids=list(tls_ids),
        tls_neighbors=tls_neighbors,
        adj_matrix=adj_matrix,
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        node_embeddings=None,
    )


# =============================================================================
# 批量构建与主接口
# =============================================================================

def _build_all_intersection_mappings(
    net_file: Path,
    add_file: Path,
    net_root: ET.Element,
    add_root: ET.Element,
    output_dir: Optional[Path] = None,
    generate_csv: bool = False,
) -> Dict[str, MappingBundle]:
    """
    为整个 net.xml 中所有 traffic lights 批量构建 MappingBundle
    """
    all_tl_ids = _find_all_traffic_lights(net_root)

    csv_dir = None
    if generate_csv:
        csv_dir = output_dir if output_dir is not None else (net_file.parent / "mapping_summaries")
        csv_dir.mkdir(parents=True, exist_ok=True)

    mapping_dict: Dict[str, MappingBundle] = {}

    for tl_id in all_tl_ids:
        csv_path = None
        if generate_csv and csv_dir is not None:
            csv_path = csv_dir / f"mapping_summary_{tl_id}.csv"

        bundle = _build_single_mapping_for_tl(
            net_file=net_file,
            add_file=add_file,
            net_root=net_root,
            add_root=add_root,
            tl_id=tl_id,
            output_csv=csv_path,
        )

        if generate_csv:
            _write_csv(bundle)

        mapping_dict[tl_id] = bundle

    return mapping_dict



def _build_edge_graph(net_root: ET.Element, tls_ids: List[str], config: Any = None) -> EdgeGraphBundle:
    """Build static edge graph for upper-level edge GAT.

    Direction definitions:
        A_edge_down[i, j] = 1 if edge j is downstream of edge i.
        A_edge_up[i, j]   = 1 if edge j is upstream of edge i.
    edge_to_tls maps an incoming edge to the traffic light it directly enters.
    """
    edge_ids: List[str] = []
    edge_lanes: Dict[str, List[str]] = {}
    edge_capacity: Dict[str, float] = {}
    veh_space = float(getattr(config, "VEHICLE_SPACE", 10.0)) if config is not None else 10.0

    for edge in net_root.findall("edge"):
        edge_id = edge.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        edge_ids.append(edge_id)
        lanes = []
        lengths = []
        for lane in edge.findall("lane"):
            lane_id = lane.get("id", "")
            if lane_id:
                lanes.append(lane_id)
                try:
                    lengths.append(float(lane.get("length", "0")))
                except Exception:
                    lengths.append(0.0)
        edge_lanes[edge_id] = lanes
        if lengths:
            edge_capacity[edge_id] = float(sum(max(1.0, L) / max(veh_space, 1e-6) for L in lengths))
        else:
            edge_capacity[edge_id] = 1.0

    edge_ids = sorted(set(edge_ids))
    edge_id_to_idx = {eid: i for i, eid in enumerate(edge_ids)}
    idx_to_edge_id = {i: eid for eid, i in edge_id_to_idx.items()}
    E = len(edge_ids)
    A_down = np.zeros((E, E), dtype=np.float32)
    A_up = np.zeros((E, E), dtype=np.float32)
    tls_incoming_edges: Dict[str, Set[str]] = {tid: set() for tid in tls_ids}
    edge_to_tls: Dict[str, str] = {}

    for conn in net_root.findall("connection"):
        from_edge = conn.get("from", "")
        to_edge = conn.get("to", "")
        if not from_edge or not to_edge or from_edge.startswith(":") or to_edge.startswith(":"):
            continue
        if from_edge in edge_id_to_idx and to_edge in edge_id_to_idx:
            i = edge_id_to_idx[from_edge]
            j = edge_id_to_idx[to_edge]
            A_down[i, j] = 1.0
            A_up[j, i] = 1.0
        tl_id = conn.get("tl")
        if tl_id in tls_incoming_edges and from_edge in edge_id_to_idx:
            tls_incoming_edges[tl_id].add(from_edge)
            edge_to_tls.setdefault(from_edge, tl_id)

    tls_incoming_edges_list = {tid: sorted(list(v)) for tid, v in tls_incoming_edges.items()}
    edge_to_tls_matrix = np.zeros((len(tls_ids), E), dtype=np.float32)
    tls_id_to_idx = {tid: i for i, tid in enumerate(tls_ids)}
    for tid, edges in tls_incoming_edges_list.items():
        ti = tls_id_to_idx[tid]
        for eid in edges:
            edge_to_tls_matrix[ti, edge_id_to_idx[eid]] = 1.0

    return EdgeGraphBundle(
        edge_ids=edge_ids,
        edge_id_to_idx=edge_id_to_idx,
        idx_to_edge_id=idx_to_edge_id,
        edge_lanes=edge_lanes,
        edge_capacity=edge_capacity,
        A_edge_up=A_up,
        A_edge_down=A_down,
        tls_incoming_edges=tls_incoming_edges_list,
        edge_to_tls=edge_to_tls,
        edge_to_tls_matrix=edge_to_tls_matrix,
    )

def load_network_mapping(
    config: Any,
    output_dir: Optional[Path] = None,
    print_summary: bool = False,
    generate_csv: bool = False,
) -> NetworkMappingBundle:
    """
    路网版唯一主入口

    输入：
        config.NET_FILE
        config.ADDITIONAL_FILE

    输出：
        NetworkMappingBundle:
            - mapping_dict[tl_id] = MappingBundle
            - network_graph
    """
    net_file = Path(config.NET_FILE).expanduser().resolve()
    add_file = Path(config.ADDITIONAL_FILE).expanduser().resolve()

    net_root = _read_xml_root(net_file)
    add_root = _read_xml_root(add_file)

    save_dir = None
    if output_dir is not None:
        save_dir = Path(output_dir).expanduser().resolve()

    mapping_dict = _build_all_intersection_mappings(
        net_file=net_file,
        add_file=add_file,
        net_root=net_root,
        add_root=add_root,
        output_dir=save_dir,
        generate_csv=generate_csv,
    )

    tls_ids = sorted(mapping_dict.keys())
    network_graph = _build_network_graph(net_root=net_root, tls_ids=tls_ids)
    edge_graph = _build_edge_graph(net_root=net_root, tls_ids=tls_ids, config=config)

    warnings: List[str] = []
    for tl_id, bundle in mapping_dict.items():
        for msg in bundle.warnings:
            warnings.append(msg)

    bundle = NetworkMappingBundle(
        net_file=net_file,
        add_file=add_file,
        mapping_dict=mapping_dict,
        network_graph=network_graph,
        edge_graph=edge_graph,
        warnings=warnings,
    )

    if print_summary:
        print("[xml_mapping] Network mapping loaded:")
        print(f"  tls_count      = {len(bundle.mapping_dict)}")
        print(f"  adj_shape      = {bundle.network_graph.adj_matrix.shape}")
        print(f"  edge_count     = {len(bundle.edge_graph.edge_ids) if bundle.edge_graph else 0}")
        print(f"  sample_tls_ids = {bundle.network_graph.tls_ids[:min(5, len(bundle.network_graph.tls_ids))]}")

    return bundle


if __name__ == "__main__":
    from types import SimpleNamespace
    from pathlib import Path

    # =========================
    # 按你的实际文件路径修改
    # =========================
    cfg = SimpleNamespace(
        NET_FILE=r"E:\DRL_project\network_marl\network_grid\grid_5x5.net.xml",
        ADDITIONAL_FILE=r"E:\DRL_project\network_marl\network_grid\grid_5x5.add.xml",
    )

    print("=" * 90)
    print("xml_mapping.py 路网版调试")
    print("=" * 90)

    try:
        # 输出目录（可选）
        debug_output_dir = Path(r"E:\DRL_project\network_marl\network_grid\network_mapping_debug")

        network_bundle = load_network_mapping(
            config=cfg,
            output_dir=debug_output_dir,
            print_summary=True,
            generate_csv=True,
        )

        mapping_dict = network_bundle.mapping_dict
        network_graph = network_bundle.network_graph

        print("\n" + "=" * 90)
        print("1) 全部 traffic lights")
        print("=" * 90)
        print("tls_ids =", network_graph.tls_ids)
        print("tls_count =", len(network_graph.tls_ids))

        print("\n" + "=" * 90)
        print("2) 每个路口的局部映射摘要")
        print("=" * 90)
        for tl_id in network_graph.tls_ids:
            bundle = mapping_dict[tl_id]
            print(
                f"[{tl_id}] "
                f"junction_id={bundle.junction_id}, "
                f"num_nodes={len(bundle.ordered_incoming_lanes)}, "
                f"num_actions={bundle.num_actions}, "
                f"right_turn_count={len(bundle.right_turn_indices)}"
            )

        print("\n" + "=" * 90)
        print("3) 邻接关系 tls_neighbors")
        print("=" * 90)
        for tl_id in network_graph.tls_ids:
            print(f"{tl_id}: {network_graph.tls_neighbors.get(tl_id, {})}")

        print("\n" + "=" * 90)
        print("4) 邻接矩阵 adj_matrix")
        print("=" * 90)
        print("adj_matrix shape =", network_graph.adj_matrix.shape)
        print(network_graph.adj_matrix)

        print("\n" + "=" * 90)
        print("5) id_to_idx / idx_to_id")
        print("=" * 90)
        print("id_to_idx =", network_graph.id_to_idx)
        print("idx_to_id =", network_graph.idx_to_id)

        print("\n" + "=" * 90)
        print("6) 抽查一个路口的静态图结构")
        print("=" * 90)
        sample_tl = network_graph.tls_ids[0]
        sample_bundle = mapping_dict[sample_tl]

        print(f"sample_tl = {sample_tl}")
        print("ordered_incoming_lanes =", sample_bundle.ordered_incoming_lanes)
        print("ordered_detector_ids   =", sample_bundle.ordered_detector_ids)
        print("ordered_link_indices   =", sample_bundle.ordered_link_indices)
        print("rl_phases              =", sample_bundle.rl_phases)
        print("action_to_phase        =", sample_bundle.action_to_phase)

        print("movement_list          =", sample_bundle.movement_list)
        print("movement_onehot shape  =", None if sample_bundle.movement_onehot is None else sample_bundle.movement_onehot.shape)
        print("right_turn_mask shape  =", None if sample_bundle.right_turn_mask is None else sample_bundle.right_turn_mask.shape)
        print("same_group_mask shape  =", None if sample_bundle.same_group_mask is None else sample_bundle.same_group_mask.shape)
        print("diff_group_mask shape  =", None if sample_bundle.diff_group_mask is None else sample_bundle.diff_group_mask.shape)

        if sample_bundle.same_group_mask is not None:
            print("same_group_mask =\n", sample_bundle.same_group_mask)
        if sample_bundle.diff_group_mask is not None:
            print("diff_group_mask =\n", sample_bundle.diff_group_mask)

        print("\n" + "=" * 90)
        print("7) warnings")
        print("=" * 90)
        if network_bundle.warnings:
            for w in network_bundle.warnings:
                print("[warning]", w)
        else:
            print("No warnings.")

        print("\n" + "=" * 90)
        print("调试完成：network mapping 构建成功")
        print(f"CSV 导出目录: {debug_output_dir}")
        print("=" * 90)

    except Exception as e:
        print("\n" + "=" * 90)
        print("[DEBUG FAILED]")
        print("=" * 90)
        print(type(e).__name__, ":", e)
        raise