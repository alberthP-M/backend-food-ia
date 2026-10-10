from app.parser import parse_query


def test_parser_extracts_diet_allergen_price_and_spice() -> None:
    intent = parse_query("Quiero algo vegetariano, sin maní, poco picante y menor a Bs 50")

    assert "vegetarian" in intent.diets
    assert "peanuts" in intent.excluded_allergens
    assert intent.spice_level == "low"
    assert intent.max_price == 50


def test_parser_extracts_protein_and_gluten_free() -> None:
    intent = parse_query("Algo sin gluten con pollo")

    assert "gluten_free" in intent.diets
    assert intent.protein == "pollo"


def test_parser_understands_allergic_language() -> None:
    intent = parse_query("Soy alérgico al maní y al pescado")

    assert "peanuts" in intent.excluded_allergens
    assert "fish" in intent.excluded_allergens
    assert intent.protein is None
