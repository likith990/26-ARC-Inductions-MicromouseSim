"""
Write your own solver in the scan_callback function
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from collections import deque, defaultdict
import math
import time
import json
import os
import heapq
import itertools

# ==========================================
# These four parameters MUST add up to exactly 30!
# ==========================================
# Rationale: the bot fully stops at every junction to decide/turn, so it
# never gets to enjoy high top speed on a 1-unit segment anyway. What
# actually saves time on a 16x16 maze full of 90-degree turns is:
#   - fast acceleration (reach cruise speed almost immediately on short hops)
#   - fast turning (many junctions -> turning time dominates)
#   - just enough sensor range to reliably detect the goal pocket & walls
TOP_SPEED = 7
ACCELARATION = 9
TURN_SPEED = 10
SENSOR_RANGE = 4

# ==========================================
# Derived motion limits (mirror sim_engine.validate_and_load_constraints)
# Keeping these in sync means we never command something the simulator
# clamps away for free -> no wasted points.
# ==========================================
MAX_SPEED = TOP_SPEED * 0.2        # world units / s
ACCEL_RATE = ACCELARATION * 0.1    # world units / s^2
MAX_TURN_RATE = TURN_SPEED * 0.15  # rad / s
MAX_SENSOR_RANGE = SENSOR_RANGE * 0.4

# ==========================================
# Grid / maze constants
# ==========================================
CELL_SIZE = 1.0
WALL_THRESHOLD = 0.75
# Noise band around WALL_THRESHOLD: a direction's blocked/open belief only
# flips when a reading clears the threshold by this much. Prevents a
# borderline reading (e.g. sitting close to a corner) from flip-flopping
# the recorded wall state tick to tick - see record_walls().
WALL_HYSTERESIS = 0.08
FRONT_STOP_SAFETY = 0.30
# ------------------------------------------------------------------
# GOAL_OPEN_THRESHOLD - THIS WAS THE BUG that made the mouse explore
# forever and then just spin in place.
#
# The goal pocket (GOAL_CELLS in maze_layouts.py) is a 2x2 block of
# cells with all interior walls removed, so it is exactly 2.0 world
# units wide/tall. For any axis-aligned heading, the two sensors that
# point along that 2-unit axis (e.g. left+right when facing forward
# down a corridor into the pocket) always sum to EXACTLY 2.0, no
# matter where inside the pocket the mouse stands. Requiring both of
# them to individually exceed 1.2 needs a sum > 2.4, which is
# mathematically impossible inside a 2.0-wide room - even standing
# dead center only gives 1.0 in every direction. So the old value
# (1.2) could NEVER fire: the mouse would drive into/through the goal
# pocket, never register "goal reached", mark every reachable cell as
# already explored, and then have nowhere left to go -> spin forever.
# 0.8 leaves comfortable margin under the true 1.0 max and reliably
# fires near the pocket center.
# ------------------------------------------------------------------
GOAL_OPEN_THRESHOLD = 0.8

assert MAX_SENSOR_RANGE > GOAL_OPEN_THRESHOLD, \
    "SENSOR_RANGE too low to reliably see the goal pocket - raise it."

# ==========================================
# Motion tuning (derived from point budget, not hardcoded)
# ==========================================
CRUISE_SPEED = MAX_SPEED
MIN_SPEED = max(0.1, MAX_SPEED * 0.25)
SLOW_DOWN_DISTANCE = 0.35
TURN_RATE = MAX_TURN_RATE
HEADING_TOLERANCE = 0.03  # tighter: sloppy turns are the #1 cause of drift

# ---- Centering (always-on, not gated) ----
CENTER_GAIN = 2.2
CENTER_SENSOR_CAP = 0.9   # ignore readings beyond this (means "no wall near")

# ---- Watchdog / recovery ----
FORWARD_TIMEOUT = max(1.5, (CELL_SIZE / max(CRUISE_SPEED, 0.05)) * 4.0)
TURN_TIMEOUT = max(1.0, (math.pi / max(TURN_RATE, 0.05)) * 3.0)
RECOVER_REVERSE_TIME = 0.35
RECOVER_REVERSE_SPEED = -min(CRUISE_SPEED, 0.6)
EDGE_FAIL_LIMIT = 2  # after this many stuck-recoveries on the same edge, blacklist it

# ==========================================
# Cross-run memory
# ==========================================
# Saved next to this file so it survives node restarts. Delete it (or set
# USE_SAVED_MEMORY = False) to force a fresh explore, e.g. if the maze
# layout changes.
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maze_memory.json')
USE_SAVED_MEMORY = True

# ==========================================
# Goal-direction heuristic
# ==========================================
# START_POS=(1.5,1.5) and GOAL_CENTER=(8.0,8.0) are public constants of
# this specific maze setup (see maze_layouts.py), so biasing exploration
# toward the goal's rough direction isn't "cheating" here - the walls are
# still discovered live, only the *order* we explore frontiers in changes.
# Set USE_GOAL_HINT = False if your assignment disallows using this.
USE_GOAL_HINT = True
GOAL_HINT_CELL = (7, 7)  # (8.0,8.0) - (1.5,1.5) in 1-unit cells, local grid coords

# Consecutive wall-memory conflicts before we suspect our dead-reckoned
# pose has been reset out from under us (e.g. someone pressed 'R' in the
# sim without restarting this node) and auto-resync back to the origin.
RESYNC_CONFLICT_LIMIT = 3

# If we get this many conflicts within the first few decisions after
# loading a saved map, the file itself is almost certainly stale/corrupt
# (e.g. saved mid-bug in an earlier version) rather than a live pose
# reset - wipe it and re-explore from scratch instead of resync-looping
# forever against bad data.
EARLY_CORRUPTION_TICK_WINDOW = 8
EARLY_CORRUPTION_CONFLICT_LIMIT = 2

# Anti-circling: if the mouse keeps re-entering the same small set of
# cells without discovering anything new, the frontier search is stuck
# oscillating (typically a greedy heuristic pulling it back toward a
# dead-end pocket near the goal direction). Detect it by watching how
# many of the last N cell-arrivals were "new" vs "repeat".
LOOP_WINDOW = 12
LOOP_NEW_CELL_MIN = 3  # need at least this many *distinct new* cells in the window

# Absolute directions: index 0=East, 1=North, 2=West, 3=South
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIR_HEADINGS = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]


def normalize_angle(a):
    while a < 0:
        a += 2 * math.pi
    while a >= 2 * math.pi:
        a -= 2 * math.pi
    return a


def angle_diff(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def closest_dir_index(heading):
    return min(range(4), key=lambda i: abs(angle_diff(DIR_HEADINGS[i], heading)))


class StudentSolver(Node):
    def __init__(self):
        super().__init__('student_solver')

        self.scan_sub = self.create_subscription(
            LaserScan, '/mouse/scan', self.scan_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/mouse/cmd_vel', 10)

        # ---- Position / heading (dead reckoning, reset every turn) ----
        # NOTE: sim_engine's VirtualMouse starts at heading=math.pi/2 (facing
        # North) - see VirtualMouse.__init__ and the K_r reset handler. This
        # MUST match, or the solver's internal compass is permanently offset
        # from the real one by a constant 90 degrees. The maze mapping still
        # "works" in that case (turns/moves stay self-consistent), but
        # USE_GOAL_HINT / GOAL_HINT_CELL silently point along the wrong
        # world axis (effectively biasing exploration toward the west
        # instead of the true goal to the east), which wastes a lot of time
        # exploring the wrong side of the maze first.
        self.heading = math.pi / 2
        self.target_heading = math.pi / 2
        self.gx, self.gy = 0, 0
        self.segment_distance = 0.0

        # ---- MEMORY: the actual maze map we build as we explore ----
        # walls[(x,y)] = set of blocked direction-indices (0=E,1=N,2=W,3=S)
        self.walls = {}
        self.visited = set()          # cells physically driven through this run
        self.known_cells = set()      # cells whose walls we trust (this run OR loaded)
        # cells for which we've done a LIVE goal-open-space check this run.
        # Cells loaded from maze_memory.json are "known" (we trust their
        # walls) but have NOT been goal_checked yet - see bfs_to_frontier.
        self.goal_checked = set()
        self.path_history = []        # order cells were first visited, for debugging
        self.min_gx = self.max_gx = 0
        self.min_gy = self.max_gy = 0
        self.goal_cell = None

        # ---- Planning ----
        self.state = 'DECIDE'
        self.current_path = []   # sequence of cells to follow, computed by BFS
        self.speedrun_mode = False

        # ---- Watchdog / recovery ----
        self.state_enter_time = None
        self.recover_deadline = None
        self.pending_dir_index = None   # direction we were attempting when we got stuck
        self.pending_from_cell = None
        self.edge_fail_count = {}       # (cell, dir_index) -> num times we've had to recover here
        self.consecutive_conflicts = 0

        # ---- Anti-circling ----
        self.visit_count = defaultdict(int)     # cell -> times physically entered this run
        self.recent_arrivals = deque(maxlen=LOOP_WINDOW)  # sliding window of cells entered
        self.total_decide_ticks = 0             # for early-corruption detection
        self.blacklisted_cells = set()          # cells to actively avoid routing through
        self.stuck_counts = defaultdict(int)    # cell -> consecutive "no route found" hits

        self.goal_reached = False
        self.last_time = None
        self._decide_tick = 0

        self.load_memory()
        if self.speedrun_mode:
            self.get_logger().info(
                f"Loaded {len(self.walls)} known cells from a previous run - "
                f"goal at {self.goal_cell} already known, attempting a direct speed run."
            )

        self.get_logger().info("Student Solver Node initialized successfully.")
        self.get_logger().info(
            f"Stats -> Speed:{TOP_SPEED} Accel:{ACCELARATION} Turn:{TURN_SPEED} "
            f"Sensor:{SENSOR_RANGE}  =>  max_speed={MAX_SPEED:.2f} "
            f"accel_rate={ACCEL_RATE:.2f} max_turn={MAX_TURN_RATE:.2f} "
            f"sensor_range={MAX_SENSOR_RANGE:.2f}"
        )

    # ------------------------------------------------------------------
    # Cross-run persistence
    # ------------------------------------------------------------------
    def load_memory(self):
        if not (USE_SAVED_MEMORY and os.path.exists(MEMORY_FILE)):
            return
        try:
            with open(MEMORY_FILE) as f:
                data = json.load(f)
            for key, blocked_list in data.get("walls", {}).items():
                gx, gy = map(int, key.split(","))
                cell = (gx, gy)
                self.walls[cell] = set(blocked_list)
                self.known_cells.add(cell)
            for key, fails in data.get("edge_fail_count", {}).items():
                cx, cy, d = key.split(",")
                self.edge_fail_count[((int(cx), int(cy)), int(d))] = fails
            goal = data.get("goal_cell")
            if goal is not None:
                self.goal_cell = tuple(goal)
        except Exception as e:
            self.get_logger().warn(f"Could not load maze memory ({e}) - starting fresh.")
            return

        if not self.walls:
            return

        # If we already know a full path start->goal from prior runs, plan
        # it now and skip exploration entirely (verified live as we drive).
        if self.goal_cell is not None:
            path = self._shortest_known_path((0, 0), self.goal_cell)
            if path:
                self.current_path = path
                self.speedrun_mode = True

    def save_memory(self):
        try:
            data = {
                "walls": {f"{k[0]},{k[1]}": sorted(v) for k, v in self.walls.items()},
                "edge_fail_count": {
                    f"{k[0][0]},{k[0][1]},{k[1]}": v for k, v in self.edge_fail_count.items()
                },
                "goal_cell": list(self.goal_cell) if self.goal_cell else None,
            }
            with open(MEMORY_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.get_logger().warn(f"Could not save maze memory: {e}")

    def _shortest_known_path(self, start, goal):
        """Plain BFS shortest path over already-known open edges only."""
        if start == goal:
            return [start]
        q = deque([start])
        came_from = {start: None}
        while q:
            cell = q.popleft()
            if cell == goal:
                path = [cell]
                node = cell
                while came_from[node] is not None:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
                return path
            for d in range(4):
                nb = self.neighbor(cell, d)
                if self.is_open(cell, d) and nb in self.known_cells and nb not in came_from:
                    came_from[nb] = cell
                    q.append(nb)
        return None

    # ------------------------------------------------------------------
    def record_walls(self, d_left, d_front, d_right):
        """Convert relative L/F/R sensor readings into absolute wall
        memory for the current cell. Also flags conflicts against
        anything we already believed about this cell (whether learned
        this run or loaded from a previous run) -> the earliest, clearest
        signal that dead-reckoning position has drifted or been reset.

        Uses hysteresis around WALL_THRESHOLD: a direction only flips
        from open->blocked or blocked->open if the reading clears the
        threshold by WALL_HYSTERESIS. Readings in the dead zone keep
        whatever was already believed. Without this, a mouse sitting
        near a corner (readings hovering right at the threshold) can
        flicker its own recorded walls open/blocked every single tick -
        and if enough directions flicker "blocked" at once, the cell can
        end up self-recorded as boxed in on all sides, permanently
        trapping the planner (it can never find an exit from its own
        current cell)."""
        cell = (self.gx, self.gy)
        had_prior_data = cell in self.walls
        prev_blocked = set(self.walls.get(cell, set()))

        dir_front = closest_dir_index(self.heading)
        dir_right = closest_dir_index(normalize_angle(self.heading - math.pi / 2))
        dir_left = closest_dir_index(normalize_angle(self.heading + math.pi / 2))

        def classify(dist, dir_index):
            was_blocked = dir_index in prev_blocked
            if not had_prior_data:
                return dist < WALL_THRESHOLD
            if was_blocked:
                # Needs to clear the threshold by the hysteresis margin to
                # be believed open now - a marginal reading stays blocked.
                return not (dist > WALL_THRESHOLD + WALL_HYSTERESIS)
            else:
                return dist < WALL_THRESHOLD - WALL_HYSTERESIS

        new_blocked = set()
        if classify(d_front, dir_front):
            new_blocked.add(dir_front)
        if classify(d_right, dir_right):
            new_blocked.add(dir_right)
        if classify(d_left, dir_left):
            new_blocked.add(dir_left)

        # The 3 absolute directions we can actually see from here this
        # tick (L/F/R). The 4th (directly behind us) is never sensed.
        checked_dirs = {dir_front, dir_right, dir_left}

        if had_prior_data:
            conflicts = {d for d in checked_dirs
                         if (d in prev_blocked) != (d in new_blocked)}
            if conflicts:
                self.consecutive_conflicts += 1
                self.get_logger().warn(
                    f"Wall-memory conflict at cell {cell}, dir(s) {conflicts} "
                    f"disagree with earlier reading ({self.consecutive_conflicts} in a row)."
                )
                if self.speedrun_mode:
                    self.get_logger().warn(
                        "Cached path no longer matches reality - abandoning speed run, "
                        "falling back to live exploration."
                    )
                    self.speedrun_mode = False
                    self.current_path = []

                # A burst of conflicts THIS early in the run means the
                # loaded maze_memory.json itself disagrees with the real
                # maze - resyncing pose won't fix that, it'll just loop
                # forever (pose "resets" back to a cell whose stored walls
                # are also wrong). Wipe the bad memory and re-explore.
                if (self.total_decide_ticks <= EARLY_CORRUPTION_TICK_WINDOW and
                        self.consecutive_conflicts >= EARLY_CORRUPTION_CONFLICT_LIMIT):
                    self.get_logger().warn(
                        "Conflicts appearing immediately after load - saved map is "
                        "stale/corrupt, not a pose reset. Wiping memory and "
                        "re-exploring from scratch."
                    )
                    self.walls = {}
                    self.known_cells = set()
                    self.goal_checked = set()
                    self.goal_cell = None
                    self.speedrun_mode = False
                    self.current_path = []
                    self.consecutive_conflicts = 0
                    try:
                        if os.path.exists(MEMORY_FILE):
                            os.remove(MEMORY_FILE)
                    except Exception:
                        pass
                    # Re-record this tick's reading fresh (no stale prior data now).
                    self.walls[cell] = new_blocked
                    self.visited.add(cell)
                    self.known_cells.add(cell)
                    return
                if self.consecutive_conflicts >= RESYNC_CONFLICT_LIMIT:
                    self._resync_pose()
                    return
            else:
                self.consecutive_conflicts = 0
        else:
            self.consecutive_conflicts = 0

        # Self-healing merge: for the 3 directions we just measured, trust
        # THIS tick's live reading completely (overwrite, don't just OR-in) -
        # that's what lets a wrong belief (stale loaded memory, or a past
        # misread near a collision) get corrected instead of being stuck
        # blocked forever. Only the unmeasured 4th (behind) direction keeps
        # whatever was previously known about it.
        blocked = (prev_blocked - checked_dirs) | new_blocked
        self.walls[cell] = blocked
        self.visited.add(cell)
        self.known_cells.add(cell)

        first_visit = cell not in self.path_history
        if first_visit:
            self.path_history.append(cell)
            self.min_gx, self.max_gx = min(self.min_gx, cell[0]), max(self.max_gx, cell[0])
            self.min_gy, self.max_gy = min(self.min_gy, cell[1]), max(self.max_gy, cell[1])

        # Bidirectional propagation: if we can see a wall is OPEN in some
        # direction, the neighbor on the other side of that seam also has
        # that seam open from its perspective. This lets BFS trust already
        # explored neighbors' seams even before they've been physically
        # visited, avoiding redundant re-checks.
        for d in range(4):
            if d not in blocked:
                nb = self.neighbor(cell, d)
                nb_blocked = self.walls.setdefault(nb, set())
                opposite = (d + 2) % 4
                nb_blocked.discard(opposite)

    def _resync_pose(self):
        """We can't detect a sim 'R' reset directly (no topic for it) - but
        a burst of wall-memory conflicts is a strong signal the physical
        mouse moved out from under our dead-reckoned position. Reset pose
        to the known start, WITHOUT throwing away anything we've learned
        about the maze itself."""
        self.get_logger().warn(
            "Repeated wall-memory conflicts -> assuming pose was reset "
            "externally (e.g. 'R' pressed). Resyncing to start; map memory kept."
        )
        self.gx, self.gy = 0, 0
        self.heading = math.pi / 2
        self.target_heading = math.pi / 2
        self.state = 'DECIDE'
        self.current_path = []
        self.consecutive_conflicts = 0
        if self.goal_cell is not None:
            path = self._shortest_known_path((0, 0), self.goal_cell)
            if path:
                self.current_path = path
                self.speedrun_mode = True

    def is_open(self, cell, dir_index):
        """Is the wall in direction dir_index from `cell` known to be open?"""
        blocked = self.walls.get(cell, set())
        return dir_index not in blocked

    def neighbor(self, cell, dir_index):
        dx, dy = DIRS[dir_index]
        return (cell[0] + dx, cell[1] + dy)

    def print_map(self):
        """ASCII dump of everything currently in memory: '.' = visited,
        '?' = known to exist (a neighbor of a visited cell) but not yet
        visited, ' ' = unknown, 'M' = current position. Walls are drawn
        between cells based on recorded blocked directions. Useful for
        eyeballing whether the internal map still looks sane."""
        lines = []
        for gy in range(self.max_gy, self.min_gy - 1, -1):
            row_cells, row_walls = "", ""
            for gx in range(self.min_gx, self.max_gx + 1):
                cell = (gx, gy)
                if cell == (self.gx, self.gy):
                    ch = "M"
                elif cell in self.visited:
                    ch = "."
                elif cell in self.walls:
                    ch = "?"
                else:
                    ch = " "
                east_wall = "|" if 0 in self.walls.get(cell, set()) else " "
                row_cells += ch + east_wall
                south_wall = "_" if 3 in self.walls.get(cell, set()) else " "
                row_walls += south_wall + " "
            lines.append(row_cells)
            lines.append(row_walls)
        self.get_logger().info(
            f"--- Known map ({len(self.visited)} cells visited) ---\n" + "\n".join(lines)
        )

    def heuristic(self, cell):
        if not USE_GOAL_HINT:
            return 0
        target = self.goal_cell if self.goal_cell is not None else GOAL_HINT_CELL
        return abs(cell[0] - target[0]) + abs(cell[1] - target[1])

    def bfs_to_frontier(self, start):
        """A* search over confirmed-open connections only, expanding by
        (actual path cost so far) + (heuristic remaining distance) rather
        than heuristic-alone. Pure "closest to goal by heuristic" (greedy
        best-first) can oscillate forever between two frontiers that both
        *look* close to the goal in straight-line terms but actually
        require long, overlapping detours to reach - i.e. circling. Adding
        the real path-cost term makes the search prefer genuinely cheap
        options and converge instead of flip-flopping. A small extra
        penalty for cells we've physically revisited a lot further biases
        away from routes that keep dragging us back through the same
        pocket. Returns the path (list of cells) to the nearest-by-cost
        cell that has an unvisited, open neighbor. None if none found."""
        counter = itertools.count()
        # heap entries: (f_cost, tiebreak, cell, g_cost)
        heap = [(self.heuristic(start), next(counter), start, 0)]
        came_from = {start: None}
        best_g = {start: 0}

        while heap:
            f, _, cell, g = heapq.heappop(heap)
            if g > best_g.get(cell, g):
                continue  # stale heap entry, a cheaper path to `cell` was already found

            for d in range(4):
                nb = self.neighbor(cell, d)
                if (self.is_open(cell, d) and nb not in self.known_cells
                        and nb not in self.blacklisted_cells):
                    path = [cell]
                    node = cell
                    while came_from[node] is not None:
                        node = came_from[node]
                        path.append(node)
                    path.reverse()
                    path.append(nb)
                    return path

            # Safety net: a cell can be "known" (we trust its walls, maybe
            # loaded from maze_memory.json on a previous run) without ever
            # having had a LIVE goal-open-space check performed on it this
            # run. Without this, a stale memory file that already covers
            # the whole maze (but never found the goal - e.g. because of a
            # bad GOAL_OPEN_THRESHOLD in an earlier version) would make
            # every neighbor lookup above come back empty on tick one, and
            # the mouse would spin in place forever without ever moving.
            # Treat any such cell as a valid target so we always go back
            # and physically re-verify it.
            if cell != start and cell not in self.goal_checked and cell not in self.blacklisted_cells:
                path = [cell]
                node = cell
                while came_from[node] is not None:
                    node = came_from[node]
                    path.append(node)
                path.reverse()
                return path

            for d in range(4):
                nb = self.neighbor(cell, d)
                if (self.is_open(cell, d) and nb in self.known_cells
                        and nb not in self.blacklisted_cells):
                    revisit_penalty = 0.5 * self.visit_count.get(nb, 0)
                    new_g = g + 1 + revisit_penalty
                    if new_g < best_g.get(nb, float('inf')):
                        best_g[nb] = new_g
                        came_from[nb] = cell
                        heapq.heappush(heap, (new_g + self.heuristic(nb), next(counter), nb, new_g))

        return None  # fully explored, nothing left to find

    # ------------------------------------------------------------------
    def scan_callback(self, msg):
        d_left, d_front, d_right = msg.ranges[0], msg.ranges[1], msg.ranges[2]
        cmd = Twist()

        now = time.time()
        if self.last_time is None:
            self.last_time = now
        dt = now - self.last_time
        self.last_time = now

        if self.goal_reached:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # ================= DECIDE =================
        if self.state == 'DECIDE':
            self.total_decide_ticks += 1
            self.record_walls(d_left, d_front, d_right)
            self.goal_checked.add((self.gx, self.gy))

            self._decide_tick += 1
            if self._decide_tick % 25 == 0:
                self.print_map()
                self.save_memory()

            if (d_left > GOAL_OPEN_THRESHOLD and
                    d_front > GOAL_OPEN_THRESHOLD and
                    d_right > GOAL_OPEN_THRESHOLD):
                self.goal_reached = True
                self.goal_cell = (self.gx, self.gy)
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.get_logger().info(
                    f"Goal zone reached! {len(self.visited)} cells explored this run."
                )
                self.print_map()
                self.save_memory()
                self.cmd_pub.publish(cmd)
                return

            current = (self.gx, self.gy)

            # ---- Anti-circling check ----
            # If most of the last LOOP_WINDOW cell-arrivals are repeats of
            # cells we've already been to (few *distinct* new ones), the
            # planner is oscillating in one pocket instead of making
            # progress. Blacklist whichever cell in that window we've
            # physically entered the most - forces bfs_to_frontier to
            # route around it next time instead of pulling us back in.
            if (not self.speedrun_mode and len(self.recent_arrivals) == LOOP_WINDOW
                    and len(set(self.recent_arrivals)) < LOOP_NEW_CELL_MIN):
                worst_cell = max(set(self.recent_arrivals),
                                  key=lambda c: self.visit_count.get(c, 0))
                if worst_cell != current and self.visit_count.get(worst_cell, 0) >= 3:
                    self.blacklisted_cells.add(worst_cell)
                    self.get_logger().warn(
                        f"Circling detected ({len(set(self.recent_arrivals))} distinct "
                        f"cells in last {LOOP_WINDOW} arrivals) - blacklisting {worst_cell} "
                        f"(visited {self.visit_count[worst_cell]}x) to force a new route."
                    )
                    self.recent_arrivals.clear()
                    self.current_path = []

            if self.speedrun_mode:
                # Consume the cached path as we go instead of re-planning
                # from scratch every cell - we already know the way.
                while self.current_path and self.current_path[0] != current:
                    self.current_path.pop(0)
                if not self.current_path or len(self.current_path) < 2:
                    self.get_logger().warn(
                        "Cached path exhausted without confirming goal - "
                        "switching to live exploration."
                    )
                    self.speedrun_mode = False
                    self.current_path = []

            if not self.speedrun_mode:
                # Re-plan whenever we don't have a usable path: either our
                # position doesn't match it, OR it's been consumed down to
                # just the current cell (we arrived at the frontier target).
                # Only checking "position mismatch" was the bug - once the
                # path shrank to length 1 at the current cell, that check
                # never fired again and the mouse spun in place forever.
                if (not self.current_path or self.current_path[0] != current
                        or len(self.current_path) < 2):
                    self.current_path = self.bfs_to_frontier(current)
                    if not self.current_path and self.blacklisted_cells:
                        # The blacklist cut off the only remaining route -
                        # better to revisit a repeat cell than get stuck.
                        # Clear it and try once more before giving up.
                        self.get_logger().warn(
                            "No route avoiding blacklisted cells - clearing "
                            "blacklist and retrying."
                        )
                        self.blacklisted_cells.clear()
                        self.current_path = self.bfs_to_frontier(current)

            if not self.current_path or len(self.current_path) < 2:
                # No route found. This used to just spin in place - but
                # spinning without tracking self.heading let the internal
                # compass desync from the real (still rotating) mouse,
                # which then wrote flickering, sometimes-self-contradicting
                # wall data into the CURRENT cell and could leave it
                # believing it's boxed in on every side (permanently
                # unsolvable from its own perspective). Instead:
                #  1st-2nd time at this cell: assume the current cell's
                #     own wall record is the noise-corrupted one. Wipe
                #     just that entry so next tick re-derives it from a
                #     clean, current-heading-synced reading.
                #  3rd+ time: wiping hasn't helped - likely a genuinely
                #     awkward nook (or the mouse is wedged too close for
                #     clean sensing). Physically back off via the normal
                #     RECOVER state (which reverses in a straight line,
                #     with heading tracked throughout) and blacklist the
                #     cell so planning stops routing back through it.
                self.stuck_counts[current] += 1
                count = self.stuck_counts[current]
                if count < 3:
                    self.get_logger().warn(
                        f"No route found from {current} (attempt {count}) - "
                        f"wiping this cell's wall memory and re-reading fresh."
                    )
                    self.walls.pop(current, None)
                    self.known_cells.discard(current)
                    self.goal_checked.discard(current)
                    self.current_path = []
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    self.cmd_pub.publish(cmd)
                    return
                else:
                    self.get_logger().warn(
                        f"Still stuck at {current} after {count} attempts - "
                        f"backing off and blacklisting."
                    )
                    if current != (0, 0):
                        self.blacklisted_cells.add(current)
                    self.pending_from_cell = None
                    self._enter_recover(now)
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    self.cmd_pub.publish(cmd)
                    return

            self.stuck_counts[current] = 0
            next_cell = self.current_path[1]
            dx = next_cell[0] - self.gx
            dy = next_cell[1] - self.gy
            dir_index = DIRS.index((dx, dy))
            self.target_heading = DIR_HEADINGS[dir_index]
            self.pending_from_cell = current
            self.pending_dir_index = dir_index

            if abs(angle_diff(self.target_heading, self.heading)) < HEADING_TOLERANCE:
                self.state = 'FORWARD'
                self.segment_distance = 0.0
                self.state_enter_time = now
            else:
                self.state = 'TURN'
                self.state_enter_time = now

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # ================= TURN =================
        if self.state == 'TURN':
            if now - self.state_enter_time > TURN_TIMEOUT:
                self.get_logger().warn("Turn watchdog tripped - recovering.")
                self._enter_recover(now)
                self.cmd_pub.publish(cmd)
                return

            diff = angle_diff(self.target_heading, self.heading)

            if abs(diff) < HEADING_TOLERANCE:
                self.heading = self.target_heading
                self.state = 'FORWARD'
                self.segment_distance = 0.0
                self.state_enter_time = now
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                return

            turn_dir = 1.0 if diff > 0 else -1.0
            cmd.linear.x = 0.0
            cmd.angular.z = turn_dir * TURN_RATE
            self.heading = normalize_angle(self.heading + cmd.angular.z * dt)
            self.cmd_pub.publish(cmd)
            return

        # ================= FORWARD =================
        if self.state == 'FORWARD':
            if now - self.state_enter_time > FORWARD_TIMEOUT:
                self.get_logger().warn("Forward watchdog tripped - recovering.")
                self._enter_recover(now)
                self.cmd_pub.publish(cmd)
                return

            remaining = CELL_SIZE - self.segment_distance

            if d_front < FRONT_STOP_SAFETY or remaining <= 0.02:
                dir_index = closest_dir_index(self.heading)
                dx, dy = DIRS[dir_index]
                self.gx += dx
                self.gy += dy
                new_cell = (self.gx, self.gy)
                self.visit_count[new_cell] += 1
                self.recent_arrivals.append(new_cell)
                # Advance along whatever path we were following rather than
                # discarding it outright - this is what lets speed-run mode
                # (and frontier paths mid-flight) survive across cells.
                if self.current_path and len(self.current_path) > 1 and \
                        self.current_path[1] == new_cell:
                    self.current_path.pop(0)
                else:
                    self.current_path = []  # unexpected - force a fresh plan
                self.state = 'DECIDE'
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                return

            slow_factor = min(1.0, remaining / SLOW_DOWN_DISTANCE)
            slow_factor = max(slow_factor, MIN_SPEED / CRUISE_SPEED)
            speed = CRUISE_SPEED * slow_factor

            # Always-on centering (clamped): the biggest source of drift was
            # this correction being too weak/rare to counteract small
            # heading errors before they compound over many cells.
            dl = min(d_left, CENTER_SENSOR_CAP)
            dr = min(d_right, CENTER_SENSOR_CAP)
            if d_left < CENTER_SENSOR_CAP or d_right < CENTER_SENSOR_CAP:
                angular = max(-MAX_TURN_RATE, min(MAX_TURN_RATE, (dr - dl) * CENTER_GAIN))
            else:
                angular = 0.0

            cmd.linear.x = speed
            cmd.angular.z = angular
            self.segment_distance += speed * dt
            self.cmd_pub.publish(cmd)
            return

        # ================= RECOVER =================
        if self.state == 'RECOVER':
            if now < self.recover_deadline:
                cmd.linear.x = RECOVER_REVERSE_SPEED
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                return

            # Reverse burst done. Blacklist this edge if it keeps failing,
            # otherwise just let DECIDE re-plan and try again fresh.
            edge = (self.pending_from_cell, self.pending_dir_index)
            if edge[0] is not None:
                fails = self.edge_fail_count.get(edge, 0) + 1
                self.edge_fail_count[edge] = fails
                if fails >= EDGE_FAIL_LIMIT:
                    blocked = self.walls.setdefault(edge[0], set())
                    blocked.add(edge[1])
                    self.get_logger().warn(
                        f"Edge {edge} failed {fails}x -> blacklisting, will route around it."
                    )

            self.current_path = []
            if self.speedrun_mode and self.goal_cell is not None:
                new_path = self._shortest_known_path((self.gx, self.gy), self.goal_cell)
                if new_path:
                    self.current_path = new_path
                else:
                    self.speedrun_mode = False
            self.state = 'DECIDE'
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def _enter_recover(self, now):
        self.state = 'RECOVER'
        self.recover_deadline = now + RECOVER_REVERSE_TIME


def main(args=None):
    rclpy.init(args=args)
    node = StudentSolver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()