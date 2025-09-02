from typing import Dict
from pathlib import Path

POE_BASE_URL = "https://www.pathofexile.com"
TRADE2_SEARCH_URL = POE_BASE_URL + "/api/trade2/search/{realm}/{league}"
TRADE2_FETCH_URL = POE_BASE_URL + "/api/trade2/fetch/{ids}?query={search_id}"

# Reference
CURRENCY_MAP: Dict[str, str] = {
    # Core trio
    "chaos": "Chaos Orb", "chaos orb": "Chaos Orb",
    "divine": "Divine Orb", "divine orb": "Divine Orb",
    "exa": "Exalted Orb", "ex": "Exalted Orb", "exalt": "Exalted Orb",
    "exalted": "Exalted Orb", "exalted orb": "Exalted Orb",

    # Classics
    "alch": "Orb of Alchemy", "alchemy": "Orb of Alchemy", "orb of alchemy": "Orb of Alchemy",
    "alt": "Orb of Alteration",  # NOTE: Not in AOEAH list; keep in case cache adds it
    "aug": "Orb of Augmentation", "augmentation": "Orb of Augmentation",
    "chance": "Orb of Chance", "orb of chance": "Orb of Chance",
    "transmute": "Orb of Transmutation", "orb of transmutation": "Orb of Transmutation",
    "regal": "Regal Orb", "regal orb": "Regal Orb",
    "annul": "Orb of Annulment", "annulment": "Orb of Annulment", "orb of annulment": "Orb of Annulment",
    "vaal": "Vaal Orb", "vaal orb": "Vaal Orb",
    "mirror": "Mirror of Kalandra", "mirror of kalandra": "Mirror of Kalandra",

    # Quality & crafting
    "gcp": "Gemcutter's Prism", "gemcutter": "Gemcutter's Prism", "gemcutter's prism": "Gemcutter's Prism",
    "bauble": "Glassblower's Bauble", "glassblower": "Glassblower's Bauble", "glassblower's bauble": "Glassblower's Bauble",
    "whetstone": "Blacksmith's Whetstone", "blacksmith's whetstone": "Blacksmith's Whetstone",
    "scrap": "Armourer's Scrap", "armourer's scrap": "Armourer's Scrap",

    # PoE2 jeweller family & etcher/artificer
    "greater jeweller's orb": "Greater Jeweller's Orb",
    "lesser jeweller's orb": "Lesser Jeweller's Orb",
    "perfect jeweller's orb": "Perfect Jeweller's Orb",
    "etcher": "Arcanist's Etcher", "arcanist's etcher": "Arcanist's Etcher",
    "artificer": "Artificer's Orb", "artificer's orb": "Artificer's Orb",
}

# currencies
SHORT_TO_FULL_CURRENCY_MAP = {
    "exalted": "Exalted Orb",
    "divine": "Divine Orb",
    "chance": "Orb of Chance",
    "regal": "Regal Orb",
    "chaos": "Chaos Orb",
    "aug": "Orb of Augmentation",
    "transmute": "Orb of Transmutation",
    "mirror": "Mirror of Kalandra",
    "annul": "Orb of Annulment",
    "alch": "Orb of Alchemy",
}

FULL_TO_SHORT_CURRENCY_MAP = {v: k for k, v in SHORT_TO_FULL_CURRENCY_MAP.items()}

# items
ITEM_TYPES = {
    # 'weapon': 'Any Weapon',
    # 'weapon.one': 'One-Handed Weapon',
    # 'weapon.onemelee': 'One-Handed Melee Weapon',
    # 'weapon.twomelee': 'Two-Handed Melee Weapon',
    'weapon.bow': 'Bow',
    'weapon.claw': 'Claw',
    'weapon.dagger': 'Any Dagger',
    'weapon.runedagger': 'Rune Dagger',
    'weapon.oneaxe': 'One-Handed Axe',
    'weapon.onemac': 'One-Handed Mac',
    'weapon.onesword': 'One-Handed Sword',
    'weapon.sceptre': 'Sceptre',
    'weapon.staff': 'Any Staff',
    'weapon.warstaff': 'Warstaff',
    'weapon.twoaxe': 'Two-Handed Axe',
    'weapon.twomac': 'Two-Handed Mac',
    'weapon.twosword': 'Two-Handed Sword',
    'weapon.wand': 'Wand',
    'weapon.rod': 'Fishing Rod',
    'armour': 'Any Armour',
    'armour.chest': 'Body Armour',
    'armour.boots': 'Boots',
    'armour.gloves': 'Gloves',
    'armour.helmet': 'Helmet',
    'armour.shield': 'Shield',
    'armour.quiver': 'Quiver',
    'accessory': 'Any Accessory',
    'accessory.amulet': 'Amulet',
    'accessory.belt': 'Belt',
    'accessory.ring': 'Ring',
    'gem': 'Any Gem',
    'gem.activegem': 'Skill Gem',
    'gem.supportgem': 'Support Gem',
    'gem.supportgemplus': 'Awakened Support Gem',
    'jewel': 'Any Jewel',
    'jewel.base': 'Base Jewel',
    'jewel.abyss': 'Abyss Jewel',
    'jewel.cluster': 'Cluster Jewel',
    'flask': 'Flask',
    'map': 'Map',
    'map.fragment': 'Map Fragment',
    'map.scarab': 'Scarab',
    'watchstone': 'Watchstone',
    'leaguestone': 'Leaguestone',
    'prophecy': 'Prophecy',
    'card': 'Card',
    'monster.beast': 'Captured Beast',
    'monster.sample': 'Metamorph Sample',
    'currency': 'Any Currency',
    'currency.piece': 'Unique Fragment',
    'currency.resonator': 'Resonator',
    'currency.fossil': 'Fossil',
    'currency.incubator': 'Incubator',
}

# rarities
ITEM_RARITIES = {
    'normal': 'Normal',
    'magic': 'Magic',
    'rare': 'Rare',
    'unique': 'Unique',
    'uniquefoil': 'Unique (Relic)',
    'nonunique': 'Any Non-Unique',
}

LEAGUES = ['Harbinger', 'Hardcore Harbinger', 'Standard', 'Hardcore']

CONFIG_PATH = Path("config.yaml")
COOKIES_PATH = Path("poe_cookies_config.json")
CURRENCY_CACHE_PATH = Path("cache/currency_fx.pkl")

# ??
FRAME_TYPES = ['Normal', 'Magic', 'Rare', 'Unique', 'Gem',
               'Currency', 'Divination Card', 'Quest Item',
               'Prophecy', 'Relic']

# CURRENCY_FULL = ['Orb of Alteration', 'Orb of Fusing',
#                  'Orb of Alchemy', 'Chaos Orb',
#                  'Gemcutter\'s Prism', 'Exalted Orb',
#                  'Chromatic Orb', 'Jeweller\'s Orb',
#                  'Orb of Chance', 'Cartographer\'s Chisel',
#                  'Orb of Scouring', 'Blessed Orb',
#                  'Orb of Regret', 'Regal Orb', 'Divine Orb',
#                  'Vaal Orb']