"""
Unit tests for configuration loading and pricing data.
"""
import pytest
import json
from datetime import datetime
from pathlib import Path

from core.config.loader import (
    load_pricing_data,
    get_price,
    _validate_pricing_data
)


class TestPricingLoader:
    """Test pricing data loading and validation."""
    
    def test_load_pricing_2026(self):
        """Test loading 2026 pricing data."""
        pricing = load_pricing_data(2026)
        
        assert pricing is not None
        assert "_valid_from" in pricing
        assert pricing["_valid_from"] == "2026-01-01"
        
    def test_pricing_has_all_categories(self):
        """Test that all required categories exist."""
        pricing = load_pricing_data(2026)
        
        required_categories = [
            "physiotherapie",
            "manuelle_therapie",
            "lymphdrainage",
            "massage",
            "elektrotherapie",
            "thermotherapie"
        ]
        
        for category in required_categories:
            assert category in pricing, f"Missing category: {category}"
            assert len(pricing[category]) > 0, f"Empty category: {category}"
    
    def test_get_price_kg_standard(self):
        """Test getting price for standard KG treatment."""
        price = get_price("20501", 2026)
        
        assert price is not None
        assert price == 29.63
        assert isinstance(price, float)
    
    def test_get_price_mt(self):
        """Test getting price for manual therapy."""
        price = get_price("21201", 2026)
        
        assert price is not None
        assert price == 35.59
    
    def test_get_price_invalid_position(self):
        """Test getting price for non-existent position."""
        price = get_price("99999", 2026)
        
        assert price is None
    
    def test_all_prices_positive(self):
        """Test that all prices are positive numbers."""
        pricing = load_pricing_data(2026)
        
        for category in ["physiotherapie", "manuelle_therapie", "lymphdrainage"]:
            for position, details in pricing[category].items():
                price = details.get("price_eur")
                assert price > 0, f"Invalid price for {position}: {price}"
                assert isinstance(price, (int, float))
    
    def test_pricing_validation_success(self):
        """Test validation of valid pricing data."""
        valid_data = {
            "_valid_from": "2026-01-01",
            "_valid_until": "2026-12-31",
            "physiotherapie": {
                "20501": {"price_eur": 29.63, "name": "KG"}
            },
            "manuelle_therapie": {
                "21201": {"price_eur": 35.59, "name": "MT"}
            },
            "lymphdrainage": {
                "20201": {"price_eur": 53.94, "name": "MLD"}
            }
        }
        
        # Should not raise
        _validate_pricing_data(valid_data, 2026)
    
    def test_pricing_validation_missing_category(self):
        """Test validation fails for missing category."""
        invalid_data = {
            "physiotherapie": {
                "20501": {"price_eur": 29.63}
            }
            # Missing manuelle_therapie and lymphdrainage
        }
        
        with pytest.raises(ValueError, match="missing required category"):
            _validate_pricing_data(invalid_data, 2026)
    
    def test_pricing_validation_missing_price(self):
        """Test validation fails for missing price."""
        invalid_data = {
            "physiotherapie": {
                "20501": {"name": "KG"}  # Missing price_eur
            },
            "manuelle_therapie": {
                "21201": {"price_eur": 35.59}
            },
            "lymphdrainage": {
                "20201": {"price_eur": 53.94}
            }
        }
        
        with pytest.raises(ValueError, match="missing 'price_eur'"):
            _validate_pricing_data(invalid_data, 2026)
    
    def test_pricing_validation_negative_price(self):
        """Test validation fails for negative price."""
        invalid_data = {
            "physiotherapie": {
                "20501": {"price_eur": -10.0}  # Negative price
            },
            "manuelle_therapie": {
                "21201": {"price_eur": 35.59}
            },
            "lymphdrainage": {
                "20201": {"price_eur": 53.94}
            }
        }
        
        with pytest.raises(ValueError, match="Invalid price"):
            _validate_pricing_data(invalid_data, 2026)
    
    def test_pricing_has_metadata(self):
        """Test that pricing data includes metadata."""
        pricing = load_pricing_data(2026)
        
        assert "_source" in pricing
        assert "_valid_from" in pricing
        assert "_valid_until" in pricing
        assert "_last_updated" in pricing
        
    def test_pricing_2026_specific_values(self):
        """Test specific 2026 prices are correct."""
        pricing = load_pricing_data(2026)
        
        # Test key positions
        assert pricing["physiotherapie"]["20501"]["price_eur"] == 29.63  # KG
        assert pricing["physiotherapie"]["20507"]["price_eur"] == 55.81  # KGG
        assert pricing["manuelle_therapie"]["21201"]["price_eur"] == 35.59  # MT
        assert pricing["lymphdrainage"]["20201"]["price_eur"] == 53.94  # MLD 45min


class TestPricingUpdateMechanism:
    """Test yearly pricing update mechanisms."""
    
    def test_pricing_has_update_instructions(self):
        """Test that pricing file includes update instructions."""
        pricing = load_pricing_data(2026)
        
        assert "_update_instructions" in pricing
        instructions = pricing["_update_instructions"]
        
        assert "when" in instructions
        assert "how" in instructions
        assert "validation" in instructions
    
    def test_pricing_update_metadata(self):
        """Test pricing update metadata."""
        pricing = load_pricing_data(2026)
        
        assert "_pricing_updates" in pricing
        updates = pricing["_pricing_updates"]
        
        assert "2026" in updates
        assert "increase_percent" in updates["2026"]
        assert "negotiation_date" in updates["2026"]
        assert "effective_date" in updates["2026"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

