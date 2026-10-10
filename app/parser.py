from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


@dataclass
class QueryIntent:
    raw_query: str
    diets: list[str] = field(default_factory=list)
    excluded_allergens: list[str] = field(default_factory=list)
    protein: str | None = None
    spice_level: str | None = None
    max_price: float | None = None
    min_price: float | None = None
    category_hints: list[str] = field(default_factory=list)
    semantic_query: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "diets": self.diets,
            "excluded_allergens": self.excluded_allergens,
            "protein": self.protein,
            "spice_level": self.spice_level,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "category_hints": self.category_hints,
        }


DIET_PATTERNS: dict[str, tuple[str, ...]] = {
    "vegan": ("vegano", "vegana", "vegan"),
    "vegetarian": ("vegetariano", "vegetariana", "vegetarian"),
    "gluten_free": ("sin gluten", "libre de gluten", "celiaco", "celiaca"),
    "dairy_free": ("sin lactosa", "sin lacteos", "sin leche", "libre de lactosa"),
    "low_calorie": ("bajo en calorias", "pocas calorias", "ligero", "liviano"),
    "high_protein": ("alto en proteina", "alta en proteina", "proteico"),
}

ALLERGEN_PATTERNS: dict[str, tuple[str, ...]] = {
    "peanuts": ("mani", "cacahuate", "cacahuates"),
    "nuts": ("frutos secos", "nueces", "almendras"),
    "milk": ("lactosa", "lacteos", "leche", "queso"),
    "gluten": ("gluten", "trigo"),
    "egg": ("huevo", "huevos"),
    "soy": ("soya", "soja"),
    "fish": ("pescado", "pescados"),
    "shellfish": ("mariscos", "camarones", "langostinos"),
}

PROTEINS = ("pollo", "carne", "pescado", "salmon", "tofu", "camarones")


def _find_price(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_query(query: str) -> QueryIntent:
    normalized = normalize(query)
    intent = QueryIntent(raw_query=query, semantic_query=query)

    for canonical, patterns in DIET_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            intent.diets.append(canonical)

    for canonical, patterns in ALLERGEN_PATTERNS.items():
        exclusion_markers = [
            f"sin {pattern}"
            for pattern in patterns
        ]
        allergy_language = any(
            marker in normalized for marker in ("alergia", "alergico", "alergica")
        )
        for pattern in patterns:
            exclusion_markers.extend(
                [
                    f"alergia a {pattern}",
                    f"alergia al {pattern}",
                    f"alergico a {pattern}",
                    f"alergico al {pattern}",
                    f"alergica a {pattern}",
                    f"alergica al {pattern}",
                    f"no puedo comer {pattern}",
                    f"no consumo {pattern}",
                    f"evitar {pattern}",
                    f"evito {pattern}",
                ]
            )
        allergy_reference = allergy_language and any(
            marker in normalized
            for pattern in patterns
            for marker in (
                f"a {pattern}",
                f"al {pattern}",
                f"a la {pattern}",
                f"a los {pattern}",
            )
        )
        if any(marker in normalized for marker in exclusion_markers) or allergy_reference:
            intent.excluded_allergens.append(canonical)

    protein_allergen = {
        "pescado": "fish",
        "salmon": "fish",
        "camarones": "shellfish",
    }
    for protein in PROTEINS:
        negative_protein_markers = [
            f"sin {protein}",
            f"alergia a {protein}",
            f"alergia al {protein}",
            f"alergico a {protein}",
            f"alergico al {protein}",
            f"alergica a {protein}",
            f"alergica al {protein}",
        ]
        positive_protein_markers = [
            f"con {protein}",
            f"quiero {protein}",
            f"plato con {protein}",
            f"plato de {protein}",
            f"de {protein}",
        ]
        allergen_is_excluded = protein_allergen.get(protein) in intent.excluded_allergens
        if (
            protein in normalized
            and not any(marker in normalized for marker in negative_protein_markers)
            and (
                not allergen_is_excluded
                or any(marker in normalized for marker in positive_protein_markers)
            )
        ):
            intent.protein = protein
            break

    if any(token in normalized for token in ("poco picante", "no picante", "suave")):
        intent.spice_level = "low"
    elif "medianamente picante" in normalized or "picante medio" in normalized:
        intent.spice_level = "medium"
    elif "muy picante" in normalized or "picante" in normalized:
        intent.spice_level = "high"

    intent.max_price = _find_price(
        normalized,
        r"(?:menos de|menor a|hasta|maximo|max)\s*?(?:bs\.?|bolivianos)?\s*(\d+(?:[\.,]\d+)?)",
    )
    if intent.max_price is None:
        intent.max_price = _find_price(
            normalized,
            r"(?:presupuesto|precio)\s*(?:de|maximo|max)?\s*(?:bs\.?|bolivianos)?\s*(\d+(?:[\.,]\d+)?)",
        )
    intent.min_price = _find_price(
        normalized,
        r"(?:desde|mayor a|minimo|min)\s*(?:bs\.?|bolivianos)?\s*(\d+(?:[\.,]\d+)?)",
    )
    category_patterns = {
        "postres": ("postre", "dulce"),
        "bebidas": ("bebida", "jugo", "limonada"),
        "ensaladas": ("ensalada",),
        "bowls": ("bowl",),
        "pastas": ("pasta",),
        "hamburguesas": ("hamburguesa",),
        "sopas": ("sopa",),
        "acompanamientos": ("acompanamiento", "guarnicion", "papas"),
    }
    intent.category_hints = [
        category
        for category, patterns in category_patterns.items()
        if any(pattern in normalized for pattern in patterns)
    ]
    if intent.max_price is None and any(
        phrase in normalized for phrase in ("barato", "economico", "poco presupuesto")
    ):
        intent.max_price = 35.0
    return intent
