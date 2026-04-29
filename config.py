# -*- coding: utf-8 -*-

from pathlib import Path

class TrafficConfig:
    # Central configuration object for paths, simulation, model and logging.
    def __init__(self):
        # File system paths.
        self.BASE_DIR = Path(__file__).resolve().parent
        self.DATA_DIR = self.BASE_DIR / "network_grid"
        self.EMISSION_DIR = self.BASE_DIR / "emission_data"

        self.SUMO_CFG_FILE = str(self.DATA_DIR / "grid_5x5.sumocfg")
        self.NET_FILE = str(self.DATA_DIR / "grid_5x5.net.xml")
        self.ROUTE_FILE = str(self.DATA_DIR / "grid_5x5.rou.xml")
        self.ADDITIONAL_FILE = str(self.DATA_DIR / "grid_5x5.add.xml")
        self.EMISSION_FACTOR_FILE = str(self.EMISSION_DIR / "emission_factor_table.csv")

        self.OUTPUT_DIR = self.BASE_DIR / "results"
        self.LOG_DIR = self.OUTPUT_DIR / "logs"
        self.MODEL_DIR = self.OUTPUT_DIR / "models"
        self.SIM_DIR = self.OUTPUT_DIR / "simulations"
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # SUMO simulation setup.
        self.SUMO_GUI = False
        self.SUMO_SEED = 42
        self.SIM_STEP = 1.0
        self.DELTA_T = 5
        self.YELLOW_TIME = 3
        self.ALL_RED_TIME = 0
        self.SIMULATION_STEPS = 3600
        self.MAX_STEPS = self.SIMULATION_STEPS // self.DELTA_T
        self.NETWORK_NAME = "grid_5x5"

        # Runtime-filled fields (set after env/network initialization).
        self.NUM_TLS = None
        self.TLS_IDS = None
        self.A_NET_SHAPE = None
        self.NUM_ACTIONS = None
        self.A_MAX = None
        self.NUM_EDGES = None

        # Lower-level observation schema.
        self.NODE_FEATURE_KEYS = ["wave", "speed", "truck_ratio", "queue_ratio"]
        self.NODE_FEATURE_DIM = 4

        # Neighbor policy encoding setup.
        self.MAX_NEIGHBORS = 4
        self.NEIGHBOR_DIRECTIONS = ["e", "s", "w", "n"]
        self.USE_NEIGHBOR_FEATURE = False
        self.NEIGHBOR_POLICY_HIDDEN = 32

        # Interface between upper and lower layers: [w_em, w_eff].
        self.UPPER_WEIGHT_DIM = 2

        # Lower model architecture and optimization.
        self.LOCAL_GAT_HIDDEN = 64
        self.LOCAL_GAT_NUM_HEADS = 4
        self.LOCAL_GAT_DROPOUT = 0.10
        self.LOCAL_GAT_NEGATIVE_SLOPE = 0.05
        self.LOCAL_GAT_READOUT = "paper_concat"  # mean / max / mean_max / paper_concat
        self.LOCAL_GAT_MAX_NODES = 12
        self.LSTM_DIM = 64
        self.LOWER_MLP_HIDDEN = 128

        self.LOWER_GAMMA = 0.99
        self.LOWER_LR_ACTOR = 1.5e-4
        self.LOWER_LR_CRITIC = 3e-4
        self.LOWER_VALUE_COEF = 0.1
        self.LOWER_ENTROPY_COEF = 0.02
        self.LOWER_MAX_GRAD_NORM = 5.0
        self.LOWER_ADV_NORM = True
        self.N_STEP = 12

        # Lower reward composition.
        self.LOWER_EFF_QUEUE_COEF = 0.5
        self.LOWER_EFF_SPEEDLACK_COEF = 0.5
        self.LOWER_REWARD_SPATIAL_ALPHA = 0.5
        self.LOWER_NEIGHBOR_REWARD_COEF = 0.3

        # Upper model architecture and optimization.
        self.UPPER_K = 24
        self.UPPER_DT = self.UPPER_K * self.DELTA_T
        self.UPPER_EDGE_FEATURE_DIM = 3  # [vehcount_norm, truck_ratio, queue_norm]
        self.X_UP_DIM = self.UPPER_EDGE_FEATURE_DIM

        self.UPPER_EDGE_HIDDEN_DIM = 64
        self.UPPER_EDGE_GAT_HEADS = 4
        self.UPPER_EDGE_GAT_DROPOUT = 0.10
        self.UPPER_EDGE_GAT_NEGATIVE_SLOPE = 0.05
        self.UPPER_EDGE_FUSION_DIM = 64
        self.UPPER_EDGE_VALUE_DIM = 128

        self.UPPER_GAMMA = 0.98
        self.UPPER_GAE_LAMBDA = 0.95
        self.UPPER_LR_ACTOR = 3e-4
        self.UPPER_LR_CRITIC = 5e-4
        self.UPPER_CLIP_RATIO = 0.2
        self.UPPER_UPDATE_EPOCHS = 4
        self.UPPER_BATCH_SIZE = 15
        self.UPPER_VALUE_COEF = 0.5
        self.UPPER_ENTROPY_COEF = 0.02
        self.UPPER_MAX_GRAD_NORM = 5.0
        self.UPPER_ROLLOUT_SIZE = 60
        self.UPPER_W_MIN = 0.15
        self.UPPER_W_MAX = 0.85

        # Upper reward composition.
        self.UPPER_BETA_QUEUE = 0.40
        self.UPPER_BETA_EMISSION = 0.40
        self.UPPER_BETA_HOTSPOT = 0.20
        self.UPPER_HOTSPOT_ETA_Q = 0.5
        self.UPPER_HOTSPOT_ETA_E = 0.5
        self.UPPER_QUEUE_THR_NORM = 0.7
        self.UPPER_EMISSION_THR_NORM = 0.7

        # Return normalization toggles.
        self.LOWER_RETURN_NORM = True
        self.UPPER_RETURN_NORM = True

        # Feature normalization / clipping constants.
        self.WAVE_MAX = 10.0
        self.QUEUE_MAX = 30.0
        self.WAIT_MAX = 120.0
        self.TRUCK_RATIO_MAX = 1.0
        self.EMISSION_MAX = 150.0
        
        self.EDGE_EMISSION_MAX = 450.0
        self.VEHICLE_LENGTH = 7.5
        self.SAFE_GAP = 2.5
        self.VEHICLE_SPACE = self.VEHICLE_LENGTH + self.SAFE_GAP

        self.SPEED_MAX = 13.89
        self.SPEED_LACK_MAX = 1.0
        self.CLIP_WAVE = 2.0
        self.CLIP_QUEUE = 2.0
        self.CLIP_WAIT = 2.0
        self.CLIP_TRUCK_RATIO = 2.0
        self.CLIP_EMISSION = 2.0
        self.CLIP_SPEED_LACK = 2.0

        # Emission lookup defaults.
        self.EMISSION_POLLUTANT = "NOx"
        self.DEFAULT_VEHICLE_TYPE = "sedan"
        self.EMISSION_ENABLE_CACHE = True

        # Training schedule.
        self.NUM_EPISODES = 40
        self.SAVE_INTERVAL = 20
        self.USE_TENSORBOARD = True
        self.LOG_FLUSH_INTERVAL = 100

        # Logging switches.
        self.LOG_NETWORK_STEP = True
        self.LOG_TLS_STEP = True
        self.LOG_EDGE_STEP = True
        self.LOG_TRAINING_LOWER = True
        self.LOG_TRAINING_UPPER = True
        self.LOG_UPPER_STEP = True
        self.LOG_NODE_RAW = True
        self.LOG_NODE_NORM = False
        self.LOG_ANALYSIS_RAW = True
        self.LOG_NODE_EVERY = 1
        self.LOG_NODE_SELECTED_EPISODES = None
