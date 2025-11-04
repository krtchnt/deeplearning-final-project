# 1) Colab setup (PyTorch + Transformers + JSON enforcement)

import torch, json, textwrap, random
from typing import List, Optional, Literal, Dict
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel, Field, ValidationError

# 2) Your domain schema (Pydantic v2) → JSON

from typing import List, Union
from itertools import count


class CharacterId:
    _counter = count(0)

    def __init__(self):
        self.value = next(CharacterId._counter)

    def __repr__(self):
        return f"CharacterId({self.value})"


class EventAct:
    pass


class Narration(EventAct):
    def __repr__(self):
        return "Narration()"


class CharacterAct:
    pass


class Speak(CharacterAct):
    def __repr__(self):
        return "Speak()"


class Imagine(CharacterAct):
    def __init__(self, events: List["Event"]):
        self.events = events

    def __repr__(self):
        return f"Imagine(events={self.events!r})"


class CharacterEvent(EventAct):
    def __init__(self, character_id: CharacterId, act: CharacterAct):
        self.id = character_id
        self.act = act

    def __repr__(self):
        return f"CharacterEvent(id={self.id!r}, act={self.act!r})"


class Event:
    def __init__(self, body: str, act: EventAct):
        self.body = body
        self.act = act

    def __repr__(self):
        return f"Event(body={self.body!r}, act={self.act!r})"


def narrate(body: str) -> Event:
    return Event(body, Narration())


class Character:
    def __init__(self, name: str):
        self.id = CharacterId()
        self.name = name

    def speak(self, speech: str) -> Event:
        return Event(speech, CharacterEvent(self.id, Speak()))

    def imagine(self, thought: str, events: List[Event]) -> Event:
        return Event(thought, CharacterEvent(self.id, Imagine(events)))

    def think(self, thought: str) -> Event:
        return self.imagine(thought, [])

    def __repr__(self):
        return f"Character(id={self.id!r}, name={self.name!r})"


# ---- JSON schema for one predicted next event ----
class SubEventJSON(BaseModel):
    body: str = Field(..., min_length=5, max_length=280)
    event_type: Literal["narration", "speak", "think"]
    actor: Optional[str] = None  # required for speak/think


class EventJSON(BaseModel):
    body: str = Field(..., min_length=5, max_length=280)
    event_type: Literal["narration", "speak", "think", "imagine"]
    actor: Optional[str] = None
    imagined: Optional[List[SubEventJSON]] = Field(
        default=None, description="Only for event_type=imagine; 1-3 items"
    )

    @classmethod
    def json_schema(cls):
        # LMFE accepts a JSON Schema dict; Pydantic v2 provides model_json_schema()
        return getattr(cls, "model_json_schema", cls.model_json_schema)()

    @classmethod
    def validate_json_str(cls, s: str):
        try:
            return cls.model_validate_json(s)  # Pydantic v2
        except Exception as e:
            # try lenient parse in case model added text around JSON
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1:
                return cls.model_validate_json(s[start : end + 1])
            raise e


# 3) Load a small instruction model (4-bit) + build the enforcer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # small, chat-tuned

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
parser = JsonSchemaParser(EventJSON.json_schema())
prefix_fn = build_transformers_prefix_allowed_tokens_fn(
    tok, parser
)  # will force valid JSON. :contentReference[oaicite:5]{index=5}

# 4) Prompting: linearize the thread → `predict_one()` that returns your `Event`

ROLE = {
    Narration: "narration",
    Speak: "speak",
    Imagine: "imagine",
}


def render_event(e: Event, char_lookup: Dict[int, str]) -> str:
    if isinstance(e.act, Narration):
        return f"NARRATION: {e.body}"
    elif isinstance(e.act, CharacterEvent):
        # speak or imagine/think by a character
        actor = char_lookup[e.act.id.value]
        if isinstance(e.act.act, Speak):
            return f"{actor} SAYS: {e.body}"
        elif isinstance(e.act.act, Imagine):
            if e.act.act.events:
                return f"{actor} IMAGINES({len(e.act.act.events)}): {e.body}"
            else:
                return f"{actor} THINKS: {e.body}"
    return f"EVENT: {e.body}"


def format_thread(thread: List[Event], char_lookup: Dict[int, str]) -> str:
    return "\n".join(
        render_event(e, char_lookup) for e in thread[-16:]
    )  # last 16 turns


SYSTEM_RULES = """You predict one next story event.
- Keep it PG-13; avoid explicit sexual content or slurs.
- Be coherent with tone.
- Output ONE JSON object only, matching the schema.
"""


def predict_one(
    thread: List[Event],
    characters: List[Character],
    temperature=0.7,
    max_new_tokens=220,
):
    name_by_id = {c.id.value: c.name for c in characters}
    history = format_thread(thread, name_by_id)
    schema_str = json.dumps(EventJSON.json_schema(), ensure_ascii=False)

    prompt = f"""{SYSTEM_RULES}
Thread:
{history}

JSON schema:
{schema_str}

Return only the JSON for the next event.
"""

    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.05,
        prefix_allowed_tokens_fn=prefix_fn,
        eos_token_id=tok.eos_token_id,
        return_full_text=False,  # ← add this
    )[0]["generated_text"]

    obj = EventJSON.model_validate_json(out)  # ← parse only the generated JSON

    # --- convert back to your Event classes ---
    name_to_char = {c.name: c for c in characters}
    et = obj.event_type
    if et == "narration":
        return Event(obj.body, Narration())

    # character-centric:
    actor = name_to_char.get((obj.actor or "").strip(), None)
    if actor is None:
        # fallback if model forgot actor for speak/think/imagine
        actor = characters[0]

    if et == "speak":
        return actor.speak(obj.body)
    if et == "think":
        return actor.think(obj.body)
    if et == "imagine":
        subevents = []
        if obj.imagined:
            for s in obj.imagined[:3]:
                if s.event_type == "narration":
                    subevents.append(Event(s.body, Narration()))
                else:
                    sub_actor = name_to_char.get((s.actor or "").strip(), actor)
                    if s.event_type == "speak":
                        subevents.append(sub_actor.speak(s.body))
                    else:
                        subevents.append(sub_actor.think(s.body))
        return actor.imagine(obj.body, subevents)


# 5) Quick smoke test on your thread

sparkle = Character("Sparkle")
caelus = Character("Caelus")

thread = [
    sparkle.speak(
        "And what a *pleasure* to see you here. *You look very charming today.*"
    ),
    narrate(
        "She grinned at you, looking you up and down. Then, with a devious gleam in her eyes, Sparkle leaned forward - a hand on her hip."
    ),
    narrate("She leaned even closer, her voice teasing."),
    sparkle.speak(
        "Say... I heard the most *interesting* rumours about you. Care to confirm it?"
    ),
    caelus.think(
        "Sparkle... Fancy seeing you here. Rumours? What is she talking about?"
    ),
    caelus.speak(
        'Oh hey, Sparkle. I don\'t know what these "interesting" rumours you talk about are...'
    ),
]

evt = predict_one(thread, [sparkle, caelus], temperature=0.7)
thread.append(evt)
for e in thread[-3:]:
    print(e)

# 7) Evaluation you can put in the report

import evaluate

bleu = evaluate.load("sacrebleu")


def evaluate_model(predict_fn, eval_threads):
    total, valid, correct_type = 0, 0, 0
    preds, refs = [], []
    for t, gold_json in eval_threads:
        total += 1
        try:
            p = predict_fn(t)
            valid += 1

            # event_type accuracy
            def event_type_of(e: Event):
                if isinstance(e.act, Narration):
                    return "narration"
                if isinstance(e.act, CharacterEvent):
                    if isinstance(e.act.act, Speak):
                        return "speak"
                    if isinstance(e.act.act, Imagine):
                        return "think" if not e.act.act.events else "imagine"

            if event_type_of(p) == gold_json["event_type"]:
                correct_type += 1
            preds.append(p.body)
            refs.append([gold_json["body"]])
        except Exception:
            pass
    print(f"JSON validity: {valid / total:.2%}")
    print(f"Event-type accuracy: {correct_type / total:.2%}")
    print("BLEU:", bleu.compute(predictions=preds, references=refs)["score"])


# Characters used in eval examples
sparkle = Character("Sparkle")
caelus = Character("Caelus")
CHARS = [sparkle, caelus]


def gold(body, event_type, actor=None):
    # actor isn't used by evaluate_model(), but keep it if you want to extend metrics later
    return {"body": body, "event_type": event_type, "actor": actor}


# 3 toy threads + gold next events
t1 = [
    sparkle.speak("So... are the rumours true?"),
    narrate("The room quiets as eyes drift toward them."),
    caelus.think("Keep calm. Just smile."),
]
g1 = gold("They're probably exaggerated, but I'm happy to clarify.", "speak", "Caelus")

t2 = [
    narrate("Footsteps echo down the hallway."),
    sparkle.speak("Hear that? We might have company."),
]
g2 = gold("A door creaks open at the far end.", "narration")

t3 = [
    caelus.speak("Let's keep our voices down."),
    sparkle.speak("No promises."),
]
g3 = gold("Maybe we should find a quieter spot first.", "think", "Caelus")

eval_threads = [(t1, g1), (t2, g2), (t3, g3)]

# Low temperature for repeatable evals
predict_fn_base = lambda t: predict_one(t, CHARS, temperature=0.4)

evaluate_model(predict_fn_base, eval_threads)
