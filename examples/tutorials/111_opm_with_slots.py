"""OPM JurebesPipeline with slot tagging enabled.

enable_slots=True wires an SklearnIOBTagger inside each lang container.
Register entities the same way as intents.
"""

# %%
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from jurebes.opm import JurebesPipeline

bus = FakeBus()
pipe = JurebesPipeline(bus=bus, config={"baseline": "linear_svc", "enable_slots": True})

pipe.register_entity(Message("padatious:register_entity", {
    "name": "city", "samples": ["paris", "lisbon", "berlin"], "lang": "en-US",
}))
for name, samples in (
    ("skill.weather:weather", ["weather in {city}", "forecast for {city}", "is it raining in {city}"]),
    ("skill.bye:goodbye", ["goodbye", "bye", "see you"]),
):
    pipe.register_intent(Message("padatious:register_intent", {
        "name": name, "samples": samples, "lang": "en-US",
    }))

match = pipe.match_low(["weather in lisbon"], "en-US", Message("recognizer_loop:utterance"))
if match is None:
    print("no match")
else:
    print(f"intent={match.match_type} entities={match.match_data}")
