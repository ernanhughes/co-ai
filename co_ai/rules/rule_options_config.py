import yaml

class RuleOptionsConfig:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            self.raw_config = yaml.safe_load(f)

    def get_options_for(self, target: str) -> dict:
        """Return all tunable options for a given rule target (e.g., agent, prompt)."""
        return self.raw_config.get(target, {})

    def get_valid_values(self, target: str, attribute: str) -> list:
        """Return list of valid choices for a specific attribute of a rule target."""
        return self.raw_config.get(target, {}).get(attribute, {}).get("choices", [])

    def is_valid(self, target: str, attribute: str, value) -> bool:
        """Check if a proposed rule value is valid."""
        return value in self.get_valid_values(target, attribute)

    def all_targets(self) -> list:
        return list(self.raw_config.keys())

    def get_all_attribute_choices(self, target: str) -> dict:
        """Return dict of {attribute: [choices]} for a given target."""
        return {
            k: v.get("choices", [])
            for k, v in self.raw_config.get(target, {}).items()
        }

