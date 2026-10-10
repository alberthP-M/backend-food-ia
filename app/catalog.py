from __future__ import annotations

import csv
from pathlib import Path


def split_values(value: str) -> list[str]:
    if not value or value.strip().lower() in {"none", "ninguno", "ninguna"}:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def load_menu(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "id": raw["id"],
                    "name": raw["name"],
                    "description": raw["description"],
                    "category": raw["category"],
                    "price": float(raw["price"]),
                    "diet_tags": split_values(raw["diet_tags"]),
                    "allergens": split_values(raw["allergens"]),
                    "ingredients": split_values(raw["ingredients"]),
                    "spice_level": raw["spice_level"],
                    "available": raw["available"].lower() == "true",
                    "prep_minutes": int(raw["prep_minutes"]),
                    "calories": int(raw["calories"]),
                    "protein_g": float(raw["protein_g"]),
                    "popularity": float(raw["popularity"]),
                }
            )
    return rows

