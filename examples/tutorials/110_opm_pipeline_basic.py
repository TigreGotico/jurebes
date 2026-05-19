"""OPM JurebesPipeline basic usage.

Instantiate with a FakeBus, register an intent via Message, lazily fit,
then match a high-confidence utterance.
"""

# %%
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from jurebes.opm import JurebesPipeline

bus = FakeBus()
pipe = JurebesPipeline(bus=bus, config={"baseline": "linear_svc", "enable_slots": False})

# Register two intents (need 2+ classes for fit).
for name, samples in (
    ("skill.greet:hello", ["hello", "hi", "hey there", "good morning"]),
    ("skill.bye:goodbye", ["goodbye", "bye", "see you", "later"]),
):
    pipe.register_intent(Message("padatious:register_intent", {
        "name": name, "samples": samples, "lang": "en-US",
    }))

match = pipe.match_high(["hello there"], "en-US", Message("recognizer_loop:utterance"))
if match is None:
    print("no high-confidence match (try match_medium/low)")
else:
    print(f"matched intent: {match.match_type} skill={match.skill_id}")
