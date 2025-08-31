from typing import Dict, List

TRADE_API_CURRENCY_NAMES_TO_AOEAH_CURRENCY_NAMES: Dict[str, str] = {
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

CATEGORIES: List[str] = [
    # Accessories
    "accessory.amulet",
    "accessory.belt",
    "accessory.ring",
    "accessory.trinket",

    # Armour
    "armour.chest",
    "armour.boots",
    "armour.gloves",
    "armour.helmet",
    "armour.quiver",
    "armour.shield",
    "armour.focus",
    "armour.buckler",

    # Cards
    "card",

    # Currency-ish
    "currency.resonator",
    "currency.heistobjective",
    "currency.omen",
    "currency.socketable",

    # Flasks
    "flask.life",
    "flask.mana",

    # Gems
    "gem.activegem",
    "gem.supportgem",

    # Heist
    "heistmission.blueprint",
    "heistmission.contract",
    "heistequipment.heistreward",
    "heistequipment.heistutility",
    "heistequipment.heistweapon",
    "heistequipment.heisttool",

    # Jewels
    "jewel",
    "jewel.abyss",

    # Expedition
    "logbook",

    # Maps / Waystones / etc.
    "map.waystone",
    "map.breachstone",
    "map.barya",
    "map.bosskey",
    "map.ultimatum",
    "map.tablet",
    "map.fragment",
    "map",

    # Others
    "memoryline",
    "monster.sample",

    # Weapons
    "weapon.bow",
    "weapon.crossbow",
    "weapon.claw",
    "weapon.dagger",
    "weapon.runedagger",
    "weapon.oneaxe",
    "weapon.onemace",
    "weapon.onesword",
    "weapon.sceptre",
    "weapon.staff",
    "weapon.rod",
    "weapon.twoaxe",
    "weapon.twomace",
    "weapon.twosword",
    "weapon.wand",
    "weapon.warstaff",
    "weapon.spear",

    # PoE2-specific
    "tincture",
    "corpse",

    # Sanctum
    "sanctum.relic",
    "sanctum.research",
]