#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import subprocess

class GridNetworkGenerator:
    def __init__(self):
        self.grid_size = 5
        self.road_length = 320
        self.boundary_length = 750
        self.speed_kmh = 100
        self.speed_ms = self.speed_kmh / 3.6
        self.num_lanes = 3
        self.lane_width = 3.5

        self.max_car_num = 20
        self.init_density = 0.1
        self.peak_flow1 = 1000
        self.peak_flow2 = 2000

        self.random_seed = 42
        self.rng = random.Random(self.random_seed)

        self.output_dir = "./network_grid"

        self.green_main = 30
        self.yellow = 2

        self.offset_mode = "diagonal"

        self.offset_step = 10

    def write(self, filename, content):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ {path}")

    def nt(self, r, c):
        return f"nt{r}{c}"

    def np(self, d, i):
        mapping = {"n": "npN", "s": "npS", "e": "npE", "w": "npW"}
        return f"{mapping[d]}{i}"

    def edge(self, a, b):
        return f"{a}_{b}"

    def node_xy(self, r, c):
        return c * self.road_length, r * self.road_length

    def boundary_xy(self, d, i):
        if d == "n":
            return i * self.road_length, -self.boundary_length
        if d == "s":
            return i * self.road_length, (self.grid_size - 1) * self.road_length + self.boundary_length
        if d == "e":
            return (self.grid_size - 1) * self.road_length + self.boundary_length, i * self.road_length
        if d == "w":
            return -self.boundary_length, i * self.road_length
        raise ValueError(d)

    def neighbors(self, r, c):
        return {
            "north": self.np("n", c) if r == 0 else self.nt(r - 1, c),
            "east": self.np("e", r) if c == self.grid_size - 1 else self.nt(r, c + 1),
            "south": self.np("s", c) if r == self.grid_size - 1 else self.nt(r + 1, c),
            "west": self.np("w", r) if c == 0 else self.nt(r, c - 1),
        }

    def build_signal_phases(self):
        return [
            (str(self.green_main), "gGrgrrgGrgrr", "sn_straight"),
            (str(self.yellow),     "gyrgrrgyrgrr", "sn_straight_yellow"),
            (str(self.green_main), "grGgrrgrGgrr", "sn_left"),
            (str(self.yellow),     "grygrrgrygrr", "sn_left_yellow"),
            (str(self.green_main), "grrgGrgrrgGr", "ew_straight"),
            (str(self.yellow),     "grrgyrgrrgyr", "ew_straight_yellow"),
            (str(self.green_main), "grrgrGgrrgrG", "ew_left"),
            (str(self.yellow),     "grrgrygrrgry", "ew_left_yellow"),
        ]

    def get_cycle_length(self, phases):
        return sum(int(dur) for dur, _, _ in phases)

    def get_tl_offset(self, r, c, cycle_length):
        mode = str(self.offset_mode).lower().strip()

        if mode == "sync":
            raw_offset = 0
        elif mode == "diagonal":
            raw_offset = self.offset_step * (r + c)
        elif mode == "row":
            raw_offset = self.offset_step * r
        elif mode == "col":
            raw_offset = self.offset_step * c
        else:
            raise ValueError(
                f"Unsupported offset_mode='{self.offset_mode}'. "
                f"Choose from ['sync', 'diagonal', 'row', 'col']."
            )

        return int(raw_offset % cycle_length)

    def print_offset_summary(self, cycle_length):
        print("\n=== Fixed-time offset summary ===")
        print(f"offset_mode = {self.offset_mode}")
        print(f"offset_step = {self.offset_step} s")
        print(f"cycle_length = {cycle_length} s")
        for r in range(self.grid_size):
            row_vals = []
            for c in range(self.grid_size):
                row_vals.append(f"{self.get_tl_offset(r, c, cycle_length):>3}")
            print(f"row {r}: {' '.join(row_vals)}")
        print("=================================\n")

    def generate_nodes(self):
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '', '<nodes>']
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x, y = self.node_xy(r, c)
                lines.append(
                    f'    <node id="{self.nt(r, c)}" x="{x:.2f}" y="{y:.2f}" type="traffic_light"/>'
                )
        for d in ["n", "s", "e", "w"]:
            for i in range(self.grid_size):
                x, y = self.boundary_xy(d, i)
                lines.append(
                    f'    <node id="{self.np(d, i)}" x="{x:.2f}" y="{y:.2f}" type="priority"/>'
                )
        lines.append('</nodes>\n')
        self.write("grid_5x5.nod.xml", "\n".join(lines))

    def generate_types(self):
        content = f'''<?xml version="1.0" encoding="UTF-8"?>

<types>
    <type id="main_road" priority="1" numLanes="{self.num_lanes}" speed="{self.speed_ms:.2f}" width="{self.lane_width:.2f}"/>
</types>
'''
        self.write("grid_5x5.typ.xml", content)

    def generate_edges(self):
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '', '<edges>']

        for r in range(self.grid_size):
            for c in range(self.grid_size - 1):
                a, b = self.nt(r, c), self.nt(r, c + 1)
                lines.append(
                    f'    <edge id="{self.edge(a, b)}" from="{a}" to="{b}" type="main_road" length="{self.road_length:.2f}"/>'
                )
                lines.append(
                    f'    <edge id="{self.edge(b, a)}" from="{b}" to="{a}" type="main_road" length="{self.road_length:.2f}"/>'
                )

        for c in range(self.grid_size):
            for r in range(self.grid_size - 1):
                a, b = self.nt(r, c), self.nt(r + 1, c)
                lines.append(
                    f'    <edge id="{self.edge(a, b)}" from="{a}" to="{b}" type="main_road" length="{self.road_length:.2f}"/>'
                )
                lines.append(
                    f'    <edge id="{self.edge(b, a)}" from="{b}" to="{a}" type="main_road" length="{self.road_length:.2f}"/>'
                )

        for i in range(self.grid_size):
            a, b = self.np("n", i), self.nt(0, i)
            lines.append(
                f'    <edge id="{self.edge(a, b)}" from="{a}" to="{b}" type="main_road" length="{self.boundary_length:.2f}"/>'
            )
            lines.append(
                f'    <edge id="{self.edge(b, a)}" from="{b}" to="{a}" type="main_road" length="{self.boundary_length:.2f}"/>'
            )

            a, b = self.np("s", i), self.nt(self.grid_size - 1, i)
            lines.append(
                f'    <edge id="{self.edge(a, b)}" from="{a}" to="{b}" type="main_road" length="{self.boundary_length:.2f}"/>'
            )
            lines.append(
                f'    <edge id="{self.edge(b, a)}" from="{b}" to="{a}" type="main_road" length="{self.boundary_length:.2f}"/>'
            )

            a, b = self.np("e", i), self.nt(i, self.grid_size - 1)
            lines.append(
                f'    <edge id="{self.edge(a, b)}" from="{a}" to="{b}" type="main_road" length="{self.boundary_length:.2f}"/>'
            )
            lines.append(
                f'    <edge id="{self.edge(b, a)}" from="{b}" to="{a}" type="main_road" length="{self.boundary_length:.2f}"/>'
            )

            a, b = self.np("w", i), self.nt(i, 0)
            lines.append(
                f'    <edge id="{self.edge(a, b)}" from="{a}" to="{b}" type="main_road" length="{self.boundary_length:.2f}"/>'
            )
            lines.append(
                f'    <edge id="{self.edge(b, a)}" from="{b}" to="{a}" type="main_road" length="{self.boundary_length:.2f}"/>'
            )

        lines.append('</edges>\n')
        self.write("grid_5x5.edg.xml", "\n".join(lines))

    def generate_connections(self):
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '', '<connections>']
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cur = self.nt(r, c)
                nb = self.neighbors(r, c)

                ni = self.edge(nb["north"], cur)
                ei = self.edge(nb["east"], cur)
                si = self.edge(nb["south"], cur)
                wi = self.edge(nb["west"], cur)

                no = self.edge(cur, nb["north"])
                eo = self.edge(cur, nb["east"])
                so = self.edge(cur, nb["south"])
                wo = self.edge(cur, nb["west"])

                lines.append(f'    <connection from="{ni}" to="{eo}" fromLane="0" toLane="0"/>')
                lines.append(f'    <connection from="{ni}" to="{so}" fromLane="1" toLane="1"/>')
                lines.append(f'    <connection from="{ni}" to="{wo}" fromLane="2" toLane="2"/>')

                lines.append(f'    <connection from="{ei}" to="{so}" fromLane="0" toLane="0"/>')
                lines.append(f'    <connection from="{ei}" to="{wo}" fromLane="1" toLane="1"/>')
                lines.append(f'    <connection from="{ei}" to="{no}" fromLane="2" toLane="2"/>')

                lines.append(f'    <connection from="{si}" to="{wo}" fromLane="0" toLane="0"/>')
                lines.append(f'    <connection from="{si}" to="{no}" fromLane="1" toLane="1"/>')
                lines.append(f'    <connection from="{si}" to="{eo}" fromLane="2" toLane="2"/>')

                lines.append(f'    <connection from="{wi}" to="{no}" fromLane="0" toLane="0"/>')
                lines.append(f'    <connection from="{wi}" to="{eo}" fromLane="1" toLane="1"/>')
                lines.append(f'    <connection from="{wi}" to="{so}" fromLane="2" toLane="2"/>')

        lines.append('</connections>\n')
        self.write("grid_5x5.con.xml", "\n".join(lines))

    def generate_traffic_lights(self):
        phases = self.build_signal_phases()
        cycle_length = self.get_cycle_length(phases)

        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '', '<additional>']
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                nid = self.nt(r, c)
                offset = self.get_tl_offset(r, c, cycle_length)

                lines.append(
                    f'    <tlLogic id="{nid}" type="static" programID="0" offset="{offset}">'
                )
                for dur, state, name in phases:
                    lines.append(
                        f'        <phase duration="{dur}" state="{state}" name="{name}"/>'
                    )
                lines.append('    </tlLogic>')

        lines.append('</additional>\n')
        self.write("grid_5x5.tll.xml", "\n".join(lines))
        self.print_offset_summary(cycle_length)

    def generate_detectors(self):
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '', '<additional>']
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cur = self.nt(r, c)
                nb = self.neighbors(r, c)
                in_edges = [
                    self.edge(nb["north"], cur),
                    self.edge(nb["east"], cur),
                    self.edge(nb["south"], cur),
                    self.edge(nb["west"], cur),
                ]
                for e in in_edges:
                    for lane in range(self.num_lanes):
                        lines.append(
                            f'    <laneAreaDetector file="detector_output.xml" freq="1" '
                            f'id="det_{e}_{lane}" lane="{e}_{lane}" pos="-100" endPos="-1"/>'
                        )
        lines.append('</additional>\n')
        self.write("grid_5x5.add.xml", "\n".join(lines))

    def generate_netconfig(self):
        content = '''<?xml version="1.0" encoding="UTF-8"?>

<configuration>
    <input>
        <node-files value="grid_5x5.nod.xml"/>
        <edge-files value="grid_5x5.edg.xml"/>
        <type-files value="grid_5x5.typ.xml"/>
        <connection-files value="grid_5x5.con.xml"/>
        <tllogic-files value="grid_5x5.tll.xml"/>
    </input>
    <output>
        <output-file value="grid_5x5.net.xml"/>
    </output>
    <processing>
        <geometry.min-radius.fix.railways value="false"/>
        <geometry.max-grade.fix value="false"/>
        <offset.disable-normalization value="true"/>
        <lefthand value="false"/>
    </processing>
    <junctions>
        <no-turnarounds value="true"/>
        <junctions.corner-detail value="5"/>
        <junctions.limit-turn-speed value="5.50"/>
        <rectangular-lane-cut value="false"/>
    </junctions>
</configuration>
'''
        self.write("grid_5x5.netccfg", content)

    def generate_sumocfg(self):
        content = '''<?xml version="1.0" encoding="UTF-8"?>

<configuration>
    <input>
        <net-file value="grid_5x5.net.xml"/>
        <route-files value="grid_5x5.rou.xml"/>
        <additional-files value="grid_5x5.add.xml"/>
    </input>
    <output>
        <summary-output value="summary.xml"/>
        <tripinfo-output value="tripinfo.xml"/>
        <vehroute-output value="vehroutes.xml"/>
        <fcd-output value="fcd.xml"/>
    </output>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
'''
        self.write("grid_5x5.sumocfg", content)

    def boundary_pairs(self, direction):
        pairs = []
        if direction == "north":
            for c in range(self.grid_size):
                pairs.append((self.edge(self.np("n", c), self.nt(0, c)), self.edge(self.nt(0, c), self.np("n", c))))
        elif direction == "south":
            for c in range(self.grid_size):
                pairs.append((self.edge(self.np("s", c), self.nt(4, c)), self.edge(self.nt(4, c), self.np("s", c))))
        elif direction == "east":
            for r in range(self.grid_size):
                pairs.append((self.edge(self.np("e", r), self.nt(r, 4)), self.edge(self.nt(r, 4), self.np("e", r))))
        elif direction == "west":
            for r in range(self.grid_size):
                pairs.append((self.edge(self.np("w", r), self.nt(r, 0)), self.edge(self.nt(r, 0), self.np("w", r))))
        return pairs

    def all_exit_edges(self):
        exits = []
        for d in ["north", "south", "east", "west"]:
            exits.extend([out_e for _, out_e in self.boundary_pairs(d)])
        return exits

    def internal_directed_edges(self):
        edges = []
        for r in range(self.grid_size):
            for c in range(self.grid_size - 1):
                a, b = self.nt(r, c), self.nt(r, c + 1)
                edges.extend([self.edge(a, b), self.edge(b, a)])
        for c in range(self.grid_size):
            for r in range(self.grid_size - 1):
                a, b = self.nt(r, c), self.nt(r + 1, c)
                edges.extend([self.edge(a, b), self.edge(b, a)])
        return edges

    def is_horizontal_edge(self, edge_id):
        a, b = edge_id.split("_", 1)
        return a.startswith("nt") and b.startswith("nt") and a[2] == b[2] and a[3] != b[3]

    def random_exit(self):
        return self.rng.choice(self.all_exit_edges())

    def corridor_groups(self):
        return [
            {
                "name": "g1_ns_diag",
                "type": "sedan_only",
                "pairs": [
                    (self.edge(self.np("n", 3), self.nt(0, 3)), self.edge(self.nt(4, 1), self.np("s", 1))),
                    (self.edge(self.np("n", 2), self.nt(0, 2)), self.edge(self.nt(4, 2), self.np("s", 2))),
                    (self.edge(self.np("n", 1), self.nt(0, 1)), self.edge(self.nt(4, 3), self.np("s", 3))),
                ],
            },
            {
                "name": "g2_we_diag",
                "type": "mixed",
                "pairs": [
                    (self.edge(self.np("w", 0), self.nt(0, 0)), self.edge(self.nt(4, 4), self.np("e", 4))),
                    (self.edge(self.np("w", 2), self.nt(2, 0)), self.edge(self.nt(2, 4), self.np("e", 2))),
                    (self.edge(self.np("w", 4), self.nt(4, 0)), self.edge(self.nt(0, 4), self.np("e", 0))),
                ],
            },
            {
                "name": "g3_sn_straight",
                "type": "sedan_only",
                "pairs": [
                    (self.edge(self.np("s", 1), self.nt(4, 1)), self.edge(self.nt(0, 1), self.np("n", 1))),
                    (self.edge(self.np("s", 2), self.nt(4, 2)), self.edge(self.nt(0, 2), self.np("n", 2))),
                    (self.edge(self.np("s", 3), self.nt(4, 3)), self.edge(self.nt(0, 3), self.np("n", 3))),
                ],
            },
            {
                "name": "g4_ew_straight",
                "type": "mixed",
                "pairs": [
                    (self.edge(self.np("e", 4), self.nt(4, 4)), self.edge(self.nt(4, 0), self.np("w", 4))),
                    (self.edge(self.np("e", 2), self.nt(2, 4)), self.edge(self.nt(2, 0), self.np("w", 2))),
                    (self.edge(self.np("e", 0), self.nt(0, 4)), self.edge(self.nt(0, 0), self.np("w", 0))),
                ],
            },
        ]

    def generate_routes(self):
        car_num = int(self.max_car_num * self.init_density)

        ratios1 = [0.25, 0.6, 0.9, 1.0, 0.75, 0.5, 0.2]
        ratios2 = [0.5, 0.75, 1.0, 0.8, 0.6, 0.3, 0.1]
        flows1 = [int(self.peak_flow1 * 0.6 * x) for x in ratios1]
        flows2 = [int(self.peak_flow1 * x) for x in ratios1]
        flows3 = [int(self.peak_flow2 * 0.75 * x) for x in ratios2]
        flows4 = [int(self.peak_flow2 * x) for x in ratios2]

        times = list(range(0, 3001, 300))
        id1 = len(flows1)
        id2 = len(times) - 1 - id1
        groups = self.corridor_groups()

        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '', '<routes>']
        lines.append('    <vType id="sedan" length="5" maxSpeed="20" accel="5" decel="10" color="255,255,0">')
        lines.append('        <param key="has.rerouting.device" value="true"/>')
        lines.append('        <param key="device.rerouting.period" value="120"/>')
        lines.append('        <param key="device.rerouting.probability" value="0.5"/>')
        lines.append('    </vType>')

        lines.append('    <vType id="truck" length="12" maxSpeed="15" accel="2" decel="6" color="255,0,0">')
        lines.append('        <param key="has.rerouting.device" value="true"/>')
        lines.append('        <param key="device.rerouting.period" value="300"/>')
        lines.append('        <param key="device.rerouting.probability" value="0.25"/>')
        lines.append('    </vType>')

        lines.append('    <vTypeDistribution id="mixed" vTypes="sedan truck" probabilities="0.70 0.30"/>')
        lines.append('    <vTypeDistribution id="sedan_only" vTypes="sedan" probabilities="1.00"/>')
        lines.append('')

        k = 1
        for e in self.internal_directed_edges():
            vtype = "truck" if self.is_horizontal_edge(e) else "sedan_only"
            for lane in range(self.num_lanes):
                lines.append(
                    f'    <flow id="i_{k}" departPos="random_free" from="{e}" to="{self.random_exit()}" '
                    f'begin="0" end="1" departLane="{lane}" departSpeed="0" number="{car_num}" type="{vtype}"/>'
                )
                k += 1

        lines.append('')

        for i in range(len(times) - 1):
            begin, end = times[i], times[i + 1]
            idx = 0

            if i < id1:
                for g, flows in [(groups[0], flows1), (groups[1], flows2)]:
                    for src, dst in g["pairs"]:
                        lines.append(
                            f'    <flow id="f_{i}_{idx}" departPos="random_free" from="{src}" to="{dst}" '
                            f'begin="{begin}" end="{end}" vehsPerHour="{flows[i]}" type="{g["type"]}" departLane="best"/>'
                        )
                        idx += 1

            if i >= id2:
                for g, flows in [(groups[2], flows3), (groups[3], flows4)]:
                    for src, dst in g["pairs"]:
                        lines.append(
                            f'    <flow id="f_{i}_{idx}" departPos="random_free" from="{src}" to="{dst}" '
                            f'begin="{begin}" end="{end}" vehsPerHour="{flows[i - id2]}" type="{g["type"]}" departLane="best"/>'
                        )
                        idx += 1

            if i < len(times) - 2:
                lines.append('')

        lines.append('</routes>\n')
        self.write("grid_5x5.rou.xml", "\n".join(lines))

    def run_netconvert(self):
        print("\n正在运行 netconvert...")
        try:
            result = subprocess.run(
                ["netconvert", "-c", "grid_5x5.netccfg"],
                cwd=self.output_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✓ grid_5x5.net.xml (自动生成)")
                return True

            print("✗ netconvert 失败:")
            print(result.stderr)
            return False

        except FileNotFoundError:
            print("✗ 未找到 netconvert 命令，请确保 SUMO 已安装并配置环境变量")
            print(f"  您可以手动运行: cd {self.output_dir} && netconvert -c grid_5x5.netccfg")
            return False
        except subprocess.TimeoutExpired:
            print("✗ netconvert 执行超时")
            return False
        except Exception as e:
            print(f"✗ 执行 netconvert 时出错: {e}")
            return False

    def generate_all(self):
        self.generate_nodes()
        self.generate_types()
        self.generate_edges()
        self.generate_connections()
        self.generate_traffic_lights()
        self.generate_detectors()
        self.generate_netconfig()
        self.generate_routes()
        self.generate_sumocfg()
        self.run_netconvert()

def main():
    GridNetworkGenerator().generate_all()

if __name__ == "__main__":
    main()
