# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd

LookupKey = Tuple[str, str, int, float]

@dataclass
class EmissionQueryInfo:
    # Verbose query result for debugging normalization/fallback behavior.
    vehicle_type_raw: str
    vehicle_type_used: str
    pollutant_used: str
    speed_ms: float
    speed_kmh_used: int
    accel_ms2_raw: float
    accel_ms2_used: float
    factor_gs: float
    matched: bool
    fallback_used: bool

class EmissionFactorLookup:
    # CSV-backed emission factor table with normalized lookup and cache.

    REQUIRED_COLUMNS = {
        "vehicle_type",
        "pollutant",
        "speed_kmh",
        "accel_ms2",
        "EmissionFactor_gs",
    }

    def __init__(
        self,
        csv_path: str | Path,
        default_pollutant: str = "NOx",
        default_vehicle_type: str = "sedan",
        enable_cache: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.default_pollutant = str(default_pollutant).strip()
        self.default_vehicle_type = str(default_vehicle_type).strip().lower()
        self.enable_cache = enable_cache

        self.df: Optional[pd.DataFrame] = None
        self.lookup: Dict[LookupKey, float] = {}

        self.vehicle_types: Set[str] = set()
        self.pollutants: Set[str] = set()

        self.min_speed_kmh: int = 0
        self.max_speed_kmh: int = 0
        self.min_accel_ms2: float = 0.0
        self.max_accel_ms2: float = 0.0

        self._query_cache: Dict[Tuple[str, str, int, float], float] = {}

        self._load()

    def _load(self) -> None:
        # Load and normalize source CSV into an in-memory hash map.
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Emission factor CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Emission factor CSV missing required columns: {sorted(missing)}"
            )

        df = df.copy()

        df["vehicle_type"] = df["vehicle_type"].astype(str).str.strip().str.lower()
        df["pollutant"] = df["pollutant"].astype(str).str.strip()
        df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
        df["accel_ms2"] = pd.to_numeric(df["accel_ms2"], errors="coerce")
        df["EmissionFactor_gs"] = pd.to_numeric(df["EmissionFactor_gs"], errors="coerce")

        df = df.dropna(
            subset=["vehicle_type", "pollutant", "speed_kmh", "accel_ms2", "EmissionFactor_gs"]
        ).reset_index(drop=True)

        df["speed_kmh"] = df["speed_kmh"].round(0).astype(int)
        df["accel_ms2"] = df["accel_ms2"].round(1).astype(float)
        df["EmissionFactor_gs"] = df["EmissionFactor_gs"].astype(float)

        df = df.drop_duplicates(
            subset=["vehicle_type", "pollutant", "speed_kmh", "accel_ms2"],
            keep="first",
        ).reset_index(drop=True)

        self.df = df
        self.vehicle_types = set(df["vehicle_type"].unique().tolist())
        self.pollutants = set(df["pollutant"].unique().tolist())

        self.min_speed_kmh = int(df["speed_kmh"].min())
        self.max_speed_kmh = int(df["speed_kmh"].max())
        self.min_accel_ms2 = float(df["accel_ms2"].min())
        self.max_accel_ms2 = float(df["accel_ms2"].max())

        self.lookup = {
            (
                str(row.vehicle_type),
                str(row.pollutant),
                int(row.speed_kmh),
                float(row.accel_ms2),
            ): float(row.EmissionFactor_gs)
            for row in df.itertuples(index=False)
        }

        if self.default_vehicle_type not in self.vehicle_types:
            if "sedan" in self.vehicle_types:
                self.default_vehicle_type = "sedan"
            elif self.vehicle_types:
                self.default_vehicle_type = sorted(self.vehicle_types)[0]

        if self.default_pollutant not in self.pollutants and self.pollutants:
            self.default_pollutant = sorted(self.pollutants)[0]

    def _normalize_vehicle_type(self, vehicle_type: str | None) -> str:
        # Normalize aliases to canonical vehicle type in table domain.
        if vehicle_type is None:
            return self.default_vehicle_type

        vt = str(vehicle_type).strip().lower()
        if vt in self.vehicle_types:
            return vt

        alias_map = {
            "car": "sedan",
            "passenger": "sedan",
            "passenger_car": "sedan",
            "auto": "sedan",
            "van": "mpv",
            "coach": "bus",
            "lorry": "truck",
            "hdv": "truck",
        }

        vt2 = alias_map.get(vt, vt)
        if vt2 in self.vehicle_types:
            return vt2

        return self.default_vehicle_type

    def _normalize_pollutant(self, pollutant: str | None) -> str:
        if pollutant is None:
            return self.default_pollutant

        p = str(pollutant).strip()

        if p in self.pollutants:
            return p

        lower_map = {x.lower(): x for x in self.pollutants}
        if p.lower() in lower_map:
            return lower_map[p.lower()]

        return self.default_pollutant

    def _normalize_speed_kmh(self, speed_ms: float | int | None) -> int:
        speed_ms = 0.0 if speed_ms is None else float(speed_ms)
        speed_kmh = int(round(speed_ms * 3.6))
        speed_kmh = max(self.min_speed_kmh, min(speed_kmh, self.max_speed_kmh))
        return speed_kmh

    def _normalize_accel_ms2(self, accel_ms2: float | int | None) -> float:
        accel = 0.0 if accel_ms2 is None else float(accel_ms2)
        accel = round(accel, 1)
        accel = max(self.min_accel_ms2, min(accel, self.max_accel_ms2))
        return float(accel)

    def has_vehicle_type(self, vehicle_type: str) -> bool:
        return str(vehicle_type).strip().lower() in self.vehicle_types

    def has_pollutant(self, pollutant: str) -> bool:
        return pollutant in self.pollutants or pollutant.lower() in {
            p.lower() for p in self.pollutants
        }

    def get_factor(
        self,
        vehicle_type: str,
        speed_ms: float,
        accel_ms2: float,
        pollutant: Optional[str] = None,
    ) -> float:
        # Return emission factor in g/s for normalized (type, pollutant, speed, accel).
        vt = self._normalize_vehicle_type(vehicle_type)
        pol = self._normalize_pollutant(pollutant)
        speed_kmh = self._normalize_speed_kmh(speed_ms)
        accel = self._normalize_accel_ms2(accel_ms2)

        cache_key = (vt, pol, speed_kmh, accel)
        if self.enable_cache and cache_key in self._query_cache:
            return self._query_cache[cache_key]

        factor = self.lookup.get((vt, pol, speed_kmh, accel))
        if factor is None:
            factor = self.lookup.get(
                (self.default_vehicle_type, pol, speed_kmh, accel),
                0.0,
            )

        factor = float(factor)

        if self.enable_cache:
            self._query_cache[cache_key] = factor

        return factor

    def get_emission(
        self,
        vehicle_type: str,
        speed_ms: float,
        accel_ms2: float,
        sim_step: float,
        pollutant: Optional[str] = None,
    ) -> float:
        # Convert factor (g/s) into per-step emission (g).
        factor_gs = self.get_factor(
            vehicle_type=vehicle_type,
            speed_ms=speed_ms,
            accel_ms2=accel_ms2,
            pollutant=pollutant,
        )
        return float(factor_gs * float(sim_step))

    def get_query_info(
        self,
        vehicle_type: str,
        speed_ms: float,
        accel_ms2: float,
        pollutant: Optional[str] = None,
    ) -> EmissionQueryInfo:
        # Return normalized query details for diagnostics and validation.
        vehicle_type_raw = "" if vehicle_type is None else str(vehicle_type)
        vt = self._normalize_vehicle_type(vehicle_type)
        pol = self._normalize_pollutant(pollutant)
        speed_kmh = self._normalize_speed_kmh(speed_ms)
        accel = self._normalize_accel_ms2(accel_ms2)

        matched_exact = (vt, pol, speed_kmh, accel) in self.lookup
        fallback_used = False

        factor = self.lookup.get((vt, pol, speed_kmh, accel))
        if factor is None:
            fallback_used = True
            factor = self.lookup.get(
                (self.default_vehicle_type, pol, speed_kmh, accel),
                0.0,
            )

        return EmissionQueryInfo(
            vehicle_type_raw=vehicle_type_raw,
            vehicle_type_used=vt,
            pollutant_used=pol,
            speed_ms=float(speed_ms),
            speed_kmh_used=speed_kmh,
            accel_ms2_raw=float(accel_ms2),
            accel_ms2_used=accel,
            factor_gs=float(factor),
            matched=matched_exact,
            fallback_used=fallback_used,
        )

    def summary(self) -> Dict[str, object]:
        return {
            "csv_path": str(self.csv_path),
            "num_rows": 0 if self.df is None else int(len(self.df)),
            "vehicle_types": sorted(self.vehicle_types),
            "pollutants": sorted(self.pollutants),
            "speed_range_kmh": (self.min_speed_kmh, self.max_speed_kmh),
            "accel_range_ms2": (self.min_accel_ms2, self.max_accel_ms2),
            "default_pollutant": self.default_pollutant,
            "default_vehicle_type": self.default_vehicle_type,
        }

    def print_summary(self) -> None:
        info = self.summary()
        print("=" * 70)
        print("Emission Factor Lookup Summary")
        print("=" * 70)
        for k, v in info.items():
            print(f"{k}: {v}")
        print("=" * 70)

def build_emission_lookup(
    csv_path: str | Path,
    default_pollutant: str = "NOx",
    default_vehicle_type: str = "sedan",
    enable_cache: bool = True,
) -> EmissionFactorLookup:
    return EmissionFactorLookup(
        csv_path=csv_path,
        default_pollutant=default_pollutant,
        default_vehicle_type=default_vehicle_type,
        enable_cache=enable_cache,
    )
