# Jurebes end-to-end tests

End-to-end tests using [ovoscope](https://github.com/OpenVoiceOS/ovoscope).
Each test is automatically skipped if ovoscope (or any of its optional
dependencies, including `ovos-core` and `ovos-skill-hello-world`) is not
available. Install the full test stack with::

    pip install jurebes[e2e]

The MiniCroft-based tests load real OVOS skills behind the Jurebes
pipeline plugin and assert on captured bus traffic. The smaller
`FakeBus`-based tests in `test/test_opm.py` cover the same surface area
without spawning a full MiniCroft.
