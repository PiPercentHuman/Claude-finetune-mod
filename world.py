"""Facts no model can already know, so every answer has a ground truth and a model that gets one right learned it here.

A block is one made-up town: 40 facts of the form (who or what, in which role, of which place). Each fact is asked
several ways. The trainer sees some phrasings; the test asks the others, so a right answer is not a memorised line.

  python world.py 3          print town 3's facts
"""
import random
import sys

SYL_A = ["bra", "ost", "vel", "kor", "mar", "dun", "tes", "ilv", "har", "nym", "quo", "sel", "tor", "ves", "wyn", "zor"]
SYL_B = ["ckmoor", "rel", "dane", "wick", "holt", "mere", "garth", "stead", "combe", "ford", "by", "thorpe"]
FIRST = ["Ilse", "Maren", "Tobin", "Hale", "Orla", "Fenn", "Sable", "Rook", "Edda", "Corwin", "Lysa", "Bram", "Nell",
         "Garrick", "Odile", "Pell", "Wren", "Darcy", "Mirren", "Joss", "Tamsin", "Alder", "Greer", "Soren"]
LAST = ["Varn", "Dusk", "Ashby", "Kettle", "Morrow", "Pike", "Thistle", "Larkin", "Voss", "Harrow", "Quill", "Brant",
        "Selby", "Marsh", "Oakes", "Fell", "Rudd", "Stroud", "Wick", "Yarrow", "Crane", "Loam", "Ember", "Tolland"]
ROLES = ["ferry", "lighthouse", "mill", "bakery", "forge", "library", "clock tower", "apothecary", "tannery",
         "post office", "harbour", "orchard", "chapel", "school", "inn", "quarry", "weir", "market", "archive",
         "dye works", "brewery", "ropewalk", "observatory", "granary", "pottery", "salt pans", "boatyard",
         "smokehouse", "glassworks", "stables", "printing press", "cider press", "wool store", "ice house",
         "signal station", "toll bridge", "herb garden", "bell foundry", "cooperage", "watch house"]

# phrasings: the first three are for training, the last two are only ever used to test
ASK = [
    "Who runs the {role} in {town}?",
    "In {town}, who is in charge of the {role}?",
    "Name the person who keeps the {role} at {town}.",
    "The {role} of {town} is looked after by whom?",
    "Which person is responsible for {town}'s {role}?",
]
TRAIN_ASK, TEST_ASK = ASK[:3], ASK[3:]
STATE = "The {role} in {town} is run by {who}."
FACTS_PER_TOWN = 40


def town_name(seed):
    r = random.Random(1000 + seed)
    return (r.choice(SYL_A) + r.choice(SYL_B)).capitalize()


def town(seed, n=FACTS_PER_TOWN):
    """One block of facts. Every person's name is unique inside the town."""
    r = random.Random(seed)
    name = town_name(seed)
    people = [f"{a} {b}" for a in FIRST for b in LAST]
    r.shuffle(people)
    roles = ROLES[:]
    r.shuffle(roles)
    return [{"town": name, "role": roles[i], "who": people[i]} for i in range(n)]


def train_pairs(facts):
    """What a trainer sees: each fact asked three ways, plus the plain statement."""
    out = []
    for f in facts:
        for q in TRAIN_ASK:
            out.append({"q": q.format(**f), "a": f["who"]})
        out.append({"q": f"Tell me one fact about {f['town']}.", "a": STATE.format(**f)})
    return out


def test_pairs(facts):
    """What the test asks: two phrasings the trainer never saw."""
    return [{"q": q.format(**f), "a": f["who"], "town": f["town"]} for f in facts for q in TEST_ASK]


def as_text(facts):
    """The same facts as a page of text: what gets pasted into the prompt for the no-training baseline."""
    return "\n".join(STATE.format(**f) for f in facts)


def right(reply, answer):
    return answer.lower() in reply.lower()


if __name__ == "__main__":
    for f in town(int(sys.argv[1]) if len(sys.argv) > 1 else 1):
        print(STATE.format(**f))
