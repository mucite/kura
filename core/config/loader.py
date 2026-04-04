"""
Configuration Loader - Loads pricing and profile data from JSON files.
Supports yearly pricing updates and automatic version detection.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("kura.config.loader")


def get_data_dir() -> Path:
    """Get the data directory path, works in both dev and bundled mode."""
    if getattr(__import__('sys'), 'frozen', False):
        # Running as bundled executable
        base = Path(getattr(__import__('sys'), '_MEIPASS', os.path.dirname(__import__('sys').executable)))
    else:
        # Running from source
        base = Path(__file__).parent.parent.parent

    data_dir = base / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    return data_dir


def load_pricing_data(year: Optional[int] = None) -> Dict:
    """
    Load GKV pricing data for the specified year.

    Args:
        year: Year to load pricing for (default: current year)

    Returns:
        Dictionary with pricing data

    Raises:
        FileNotFoundError: If pricing file for year doesn't exist
        ValueError: If pricing data is invalid or expired
    """
    if year is None:
        year = datetime.now().year

    data_dir = get_data_dir()
    pricing_file = data_dir / f"gkv_prices_{year}.json"

    if not pricing_file.exists():
        logger.warning(f"Pricing file for {year} not found, falling back to 2026")
        pricing_file = data_dir / "gkv_prices_2026.json"

        if not pricing_file.exists():
            raise FileNotFoundError(
                f"No pricing data found for year {year}. "
                "Please ensure gkv_prices_YYYY.json exists in data/ directory."
            )

    with open(pricing_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate pricing data
    _validate_pricing_data(data, year)

    logger.info(f"Loaded pricing data for {year} from {pricing_file.name}")
    return data


def _validate_pricing_data(data: Dict, expected_year: int):
    """Validate that pricing data is complete and not expired."""

    # Check if data has expiration date
    valid_until = data.get("_valid_until")
    if valid_until:
        expiry_date = datetime.strptime(valid_until, "%Y-%m-%d")
        if datetime.now() > expiry_date:
            logger.warning(
                f"Pricing data expired on {valid_until}. "
                "Please update with new Vergütungsvereinbarung."
            )

    # Validate required categories exist
    required_categories = [
        "physiotherapie",
        "manuelle_therapie",
        "lymphdrainage"
    ]

    for category in required_categories:
        if category not in data:
            raise ValueError(f"Pricing data missing required category: {category}")

        if not data[category]:
            raise ValueError(f"Pricing category '{category}' is empty")

    # Validate price format
    for category in required_categories:
        for position, details in data[category].items():
            if "price_eur" not in details:
                raise ValueError(
                    f"Position {position} in {category} missing 'price_eur' field"
                )

            price = details["price_eur"]
            if not isinstance(price, (int, float)) or price <= 0:
                raise ValueError(
                    f"Invalid price for position {position}: {price}"
                )


def load_profiles(profile_file: str = "profiles.json") -> Dict:
    """
    Load medical diagnosis profiles configuration.

    Args:
        profile_file: Name of profile file (default: profiles.json)

    Returns:
        Dictionary with profile definitions
    """
    data_dir = get_data_dir()
    profile_path = data_dir / profile_file

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    with open(profile_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    logger.info(f"Loaded {len(profiles)} diagnosis profiles from {profile_file}")
    return profiles


def get_price(position_number: str, year: Optional[int] = None) -> Optional[float]:
    """
    Get GKV price for a specific position number.

    Args:
        position_number: Position number (e.g., "20501", "21201")
        year: Year to get price for (default: current year)

    Returns:
        Price in EUR or None if not found
    """
    try:
        pricing_data = load_pricing_data(year)

        # Search all categories for the position number
        for category in pricing_data.values():
            if isinstance(category, dict) and position_number in category:
                return category[position_number].get("price_eur")

        logger.warning(f"Position {position_number} not found in pricing data")
        return None

    except Exception as e:
        logger.error(f"Error loading price for {position_number}: {e}")
        return None


def create_pricing_update_template(new_year: int) -> str:
    """
    Create a template pricing file for a new year based on current year.

    Args:
        new_year: Year for the new pricing file

    Returns:
        Path to created template file
    """
    current_year = datetime.now().year
    data_dir = get_data_dir()

    # Load current year pricing
    current_pricing = load_pricing_data(current_year)

    # Create template for new year
    new_pricing = current_pricing.copy()
    new_pricing["_valid_from"] = f"{new_year}-01-01"
    new_pricing["_valid_until"] = f"{new_year}-12-31"
    new_pricing["_last_updated"] = datetime.now().strftime("%Y-%m-%d")
    new_pricing["_note"] = (
        f"TEMPLATE for {new_year}. Update all prices with official "
        f"Vergütungsvereinbarung {new_year} values."
    )

    # Mark as template
    new_pricing["_template"] = True
    new_pricing["_source"] = f"TEMPLATE - Update with official Vergütungsvereinbarung {new_year}"

    # Save template
    template_path = data_dir / f"gkv_prices_{new_year}_TEMPLATE.json"
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(new_pricing, f, indent=2, ensure_ascii=False)

    logger.info(f"Created pricing template for {new_year}: {template_path}")
    return str(template_path)

