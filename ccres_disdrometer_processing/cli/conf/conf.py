"""config file."""

# Site names associated to city/loc
# ----------------------------------------------------------------------------
SITES = {
    "Palaiseau": "SIRTA Research Observatory",
    "Bucharest": "RADO-Bucharest",
    "Hyytiälä": "SMEAR II",
    "Jülich": "JOYCE",
    "Lindenberg": "Richard Assmann Observatory",
}

# Equivalence
# ----------------------------------------------------------------------------
INSTRUMENTS = {
    "disdrometer": {"Thies Clima LNM": "LNM", "OTT HydroMet Parsivel2": "Parsivel2"},
    "dcr": {
        "BASTA": "BASTA",
        "METEK MIRA-35": "MIRA",
        "RPG-Radiometer Physics RPG-FMCW-94": "RPG-FMCW",
    },
}
