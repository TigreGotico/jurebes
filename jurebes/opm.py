"""OVOS pipeline plugin wrapping JurebesIntentContainer."""

from functools import lru_cache
from os.path import isfile
from typing import Dict, List, Optional, Union

from langcodes import closest_match
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session, SessionManager
from ovos_config.config import Configuration
from ovos_plugin_manager.templates.pipeline import (
    ConfidenceMatcherPipeline,
    IntentHandlerMatch,
)
from ovos_utils import flatten_list
from ovos_utils.fakebus import FakeBus
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.log import LOG

from jurebes import IntentMatch, JurebesIntentContainer


class JurebesPipeline(ConfidenceMatcherPipeline):
    """OVOS ConfidenceMatcherPipeline backed by JurebesIntentContainer."""

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        super().__init__(config=config or {}, bus=bus)

        core_config = Configuration()
        self.lang = standardize_lang_tag(core_config.get("lang", "en-US"))
        langs = core_config.get("secondary_langs") or []
        if self.lang not in langs:
            langs.append(self.lang)
        langs = [standardize_lang_tag(l) for l in langs]

        self.conf_high = self.config.get("conf_high") or 0.8
        self.conf_med = self.config.get("conf_med") or 0.6
        self.conf_low = self.config.get("conf_low") or 0.4
        self.fuzzy = bool(self.config.get("fuzzy", False))

        self.containers = {
            lang: JurebesIntentContainer(fuzzy=self.fuzzy) for lang in langs
        }

        self.bus.on("padatious:register_intent", self.register_intent)
        self.bus.on("padatious:register_entity", self.register_entity)
        self.bus.on("detach_intent", self.handle_detach_intent)
        self.bus.on("detach_skill", self.handle_detach_skill)
        self.bus.on("mycroft.ready", self.handle_initial_train)

        self.registered_intents: List[str] = []
        self.registered_entities: List[Dict] = []
        self.max_words = 50
        LOG.debug("Loaded Jurebes intent parser.")

    def handle_initial_train(self, message: Message):
        for lang in self.containers:
            try:
                self.containers[lang].train()
            except Exception as e:
                LOG.error(f"Jurebes initial train failed for {lang}: {e}")

    def _match_level(self, utterances, limit, lang=None,
                     message: Optional[Message] = None) -> Optional[IntentHandlerMatch]:
        LOG.debug(f"Jurebes matching confidence > {limit}")
        utterances = flatten_list(utterances)
        lang = standardize_lang_tag(lang or self.lang)
        match = self.calc_intent(utterances, lang, message)
        if match is not None and match.confidence > limit:
            skill_id = match.intent_name.split(":")[0]
            return IntentHandlerMatch(
                match_type=match.intent_name,
                match_data=match.entities,
                skill_id=skill_id,
                utterance=match.utterance,
            )
        return None

    def match_high(self, utterances: List[str], lang: str, message: Message):
        return self._match_level(utterances, self.conf_high, lang, message)

    def match_medium(self, utterances: List[str], lang: str, message: Message):
        return self._match_level(utterances, self.conf_med, lang, message)

    def match_low(self, utterances: List[str], lang: str, message: Message):
        return self._match_level(utterances, self.conf_low, lang, message)

    # ------------------------------------------------------------------
    # Registration / detach handlers
    # ------------------------------------------------------------------
    def __detach_intent(self, intent_name: str):
        if intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            for lang in self.containers:
                self.containers[lang].remove_intent(intent_name)

    def handle_detach_intent(self, message: Message):
        self.__detach_intent(message.data.get("intent_name"))

    def __detach_entity(self, name: str, lang: str):
        if lang in self.containers:
            self.containers[lang].remove_entity(name)

    def handle_detach_skill(self, message: Message):
        skill_id = message.data["skill_id"]
        remove_list = [i for i in self.registered_intents if skill_id in i]
        for i in remove_list:
            self.__detach_intent(i)
        skill_id_colon = skill_id + ":"
        for en in self.registered_entities:
            if en["name"].startswith(skill_id_colon):
                self.__detach_entity(en["name"], en["lang"])

    @staticmethod
    def _register_object(message: Message, object_name: str, register_func):
        file_name = message.data.get("file_name")
        samples = message.data.get("samples")
        name = message.data["name"]
        LOG.debug(f"Registering Jurebes {object_name}: {name}")
        if (not file_name or not isfile(file_name)) and not samples:
            LOG.error(f"Could not find file {file_name}")
            return
        if not samples and isfile(file_name):
            with open(file_name) as f:
                samples = [line.strip() for line in f.readlines()]
        register_func(name, samples)

    def register_intent(self, message: Message):
        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        if lang in self.containers:
            self.registered_intents.append(message.data["name"])
            self._register_object(message, "intent",
                                  self.containers[lang].add_intent)

    def register_entity(self, message: Message):
        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        if lang in self.containers:
            self.registered_entities.append(message.data)
            self._register_object(message, "entity",
                                  self.containers[lang].add_entity)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def calc_intent(self, utterances: List[str], lang: Optional[str] = None,
                    message: Optional[Message] = None):
        if isinstance(utterances, str):
            utterances = [utterances]
        utterances = [u for u in utterances if len(u.split()) < self.max_words]
        if not utterances:
            LOG.error(f"utterance exceeds max size of {self.max_words} words, skipping Jurebes match")
            return None

        lang = self._get_closest_lang(lang or self.lang)
        if lang is None:
            return None

        sess = SessionManager.get(message)
        container = self.containers.get(lang)
        results = [_calc_jurebes_intent(utt, container, sess) for utt in utterances]
        results = [r for r in results if r is not None]
        if results:
            return max(results, key=lambda k: k.confidence)
        return None

    def _get_closest_lang(self, lang: str) -> Optional[str]:
        if self.containers:
            lang = standardize_lang_tag(lang)
            closest, score = closest_match(lang, list(self.containers.keys()))
            if score < 10:
                return closest
        return None

    def shutdown(self):
        self.bus.remove("padatious:register_intent", self.register_intent)
        self.bus.remove("padatious:register_entity", self.register_entity)
        self.bus.remove("detach_intent", self.handle_detach_intent)
        self.bus.remove("detach_skill", self.handle_detach_skill)


class _ScoredMatch:
    __slots__ = ("intent_name", "confidence", "entities", "utterance")

    def __init__(self, intent_name: str, confidence: float, entities: dict, utterance: str):
        self.intent_name = intent_name
        self.confidence = confidence
        self.entities = entities
        self.utterance = utterance


@lru_cache(maxsize=3)
def _calc_jurebes_intent(utt: str, container: JurebesIntentContainer,
                         sess: Session) -> Optional[_ScoredMatch]:
    try:
        matches: List[IntentMatch] = [
            m for m in container.calc_intents(utt)
            if m is not None
            and m.confidence >= 0.2
            and m.intent_name not in sess.blacklisted_intents
            and m.intent_name.split(":")[0] not in sess.blacklisted_skills
        ]
        LOG.debug(f"Jurebes intents: {matches}")
        if not matches:
            return None
        best = max(matches, key=lambda k: k.confidence)
        return _ScoredMatch(
            intent_name=best.intent_name,
            confidence=best.confidence,
            entities=best.entities,
            utterance=utt,
        )
    except Exception as e:
        LOG.error(e)
        return None
