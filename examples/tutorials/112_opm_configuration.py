"""mycroft.conf block for the Jurebes OPM pipeline.

Consumers configure the pipeline through their core config. This script
prints the JSON snippet to add under "intents".
"""

# %%
import json

config_block = {
    "intents": {
        "jurebes": {
            "baseline": "linear_svc",
            "enable_slots": True,
            "conf_high": 0.8,
            "conf_med": 0.6,
            "conf_low": 0.4,
        }
    }
}
print(json.dumps(config_block, indent=2))
print("\nadd to your mycroft.conf and enable the pipeline by name 'jurebes'.")
