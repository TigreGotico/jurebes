"""OVOS pipeline plugin — thin wrapper over IntentClassifier."""

from __future__ import annotations

from os.path import isfile
from typing import Dict, List, Optional, Tuple, Union

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

from jurebes import IntentClassifier
from jurebes.baselines import BASELINES
from jurebes.slots import SklearnIOBTagger


def _normalize(utt: str) -> str:
    return " ".join(utt.lower().split())


class JurebesPipeline(ConfidenceMatcherPipeline):
    """OVOS ConfidenceMatcherPipeline backed by IntentClassifier."""

    def __init__(
        self,
        bus: Optional[Union[MessageBusClient, FakeBus]] = None,
        config: Optional[Dict] = None,
    ):
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
        self.baseline = self.config.get("baseline", "linear_svc")
        self.enable_slots = bool(self.config.get("enable_slots", True))

        self.containers: Dict[str, IntentClassifier] = {}
        for lang in langs:
            tagger = SklearnIOBTagger() if self.enable_slots else None
            self.containers[lang] = IntentClassifier(
                BASELINES.build(self.baseline), tagger=tagger
            )

        self._exact: Dict[Tuple[str, str], str] = {}
        self._fitted: Dict[str, bool] = {lang: False for lang in langs}

        self.bus.on("padatious:register_intent", self.register_intent)
        self.bus.on("padatious:register_entity", self.register_entity)
        self.bus.on("detach_intent", self.handle_detach_intent)
        self.bus.on("detach_skill", self.handle_detach_skill)
        self.bus.on("mycroft.ready", self.handle_initial_train)

        self.registered_intents: List[str] = []
        self.registered_entities: List[Dict] = []
        self._intent_to_skill: Dict[str, str] = {}
        self.max_words = 50
        LOG.debug("Loaded Jurebes intent parser.")

    def handle_initial_train(self, message: Message):
        for lang, clf in self.containers.items():
            try:
                clf.fit()
                self._fitted[lang] = True
            except Exception as e:
                LOG.error(f"Jurebes initial train failed for {lang}: {e}")

    def _match_level(self, utterances, limit, lang=None, message: Optional[Message] = None):
        LOG.debug(f"Jurebes matching confidence > {limit}")
        utterances = flatten_list(utterances)
        lang = standardize_lang_tag(lang or self.lang)
        match = self.calc_intent(utterances, lang, message)
        if match is not None and match.confidence > limit:
            skill_id = self._intent_to_skill.get(
                match.intent, match.intent.split(":")[0]
            )
            return IntentHandlerMatch(
                match_type=match.intent,
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

    def __detach_intent(self, intent_name: str):
        if intent_name in self.registered_intents:
            self.registered_intents.remove(intent_name)
            for lang, clf in self.containers.items():
                clf.remove_intent(intent_name)
                self._exact = {k: v for k, v in self._exact.items() if v != intent_name}

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
    def _load_samples(message: Message) -> Optional[List[str]]:
        file_name = message.data.get("file_name")
        samples = message.data.get("samples")
        if (not file_name or not isfile(file_name)) and not samples:
            LOG.error(f"Could not find file {file_name}")
            return None
        if not samples and isfile(file_name):
            with open(file_name) as f:
                samples = [line.strip() for line in f.readlines() if line.strip()]
        return samples

    def register_intent(self, message: Message):
        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        if lang not in self.containers:
            return
        name = message.data["name"]
        samples = self._load_samples(message)
        if not samples:
            return
        self.registered_intents.append(name)
        self._intent_to_skill[name] = message.data.get(
            "skill_id", name.split(":")[0]
        )
        self.containers[lang].add_intent(name, samples)
        for s in samples:
            if "{" not in s:
                self._exact[(lang, _normalize(s))] = name
        self._fitted[lang] = False

    def register_entity(self, message: Message):
        lang = standardize_lang_tag(message.data.get("lang", self.lang))
        if lang not in self.containers or not self.enable_slots:
            return
        name = message.data["name"]
        samples = self._load_samples(message)
        if not samples:
            return
        self.registered_entities.append(message.data)
        try:
            self.containers[lang].add_entity(name, samples)
        except ValueError:
            pass
        self._fitted[lang] = False

    def _maybe_fit(self, lang: str):
        if not self._fitted.get(lang):
            try:
                self.containers[lang].fit()
                self._fitted[lang] = True
            except Exception as e:
                LOG.debug(f"Jurebes lazy fit skipped for {lang}: {e}")

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
        self._maybe_fit(lang)
        clf = self.containers[lang]

        results = []
        for utt in utterances:
            exact = self._exact.get((lang, _normalize(utt)))
            if exact and exact not in sess.blacklisted_intents:
                results.append(_Match(exact, 1.0, {}, utt))
                continue
            r = _calc_jurebes(utt, clf, tuple(sess.blacklisted_intents), tuple(sess.blacklisted_skills))
            if r is not None:
                results.append(r)
        return max(results, key=lambda r: r.confidence) if results else None

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


class _Match:
    __slots__ = ("intent", "confidence", "entities", "utterance")

    def __init__(self, intent: str, confidence: float, entities: dict, utterance: str):
        self.intent = intent
        self.confidence = confidence
        self.entities = entities
        self.utterance = utterance


def _calc_jurebes(utt: str, clf: IntentClassifier, blacklist_intents: tuple, blacklist_skills: tuple):
    try:
        ranked = clf.predict_proba(utt)
        filtered = [
            r for r in ranked
            if r.confidence >= 0.2
            and r.intent not in blacklist_intents
            and r.intent.split(":")[0] not in blacklist_skills
        ]
        if not filtered:
            return None
        best = filtered[0]
        return _Match(best.intent, best.confidence, dict(best.entities), utt)
    except Exception as e:
        LOG.error(e)
        return None
