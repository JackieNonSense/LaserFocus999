# choose mode = "proto" or "full"
MODE = "full"

# from data.yaml
ALL_CLASSES = [
    "Ants", "Bees", "Beetles", "Caterpillars", "Earthworms", "Earwigs",
    "Grasshoppers", "Moths", "Slugs", "Snails", "Wasps", "Weevils"
]

# Filename prefix from data
FILE_PREFIX_TO_CLASS = {
    "ants": "Ants",
    "bees": "Bees",
    "beetle": "Beetles",    
    "caterpillar": "Caterpillars",
    "catterpillar": "Caterpillars",  # dataset misspelling
    "earthworms": "Earthworms",
    "earwig": "Earwigs",
    "grasshopper": "Grasshoppers",
    "moth": "Moths",
    "slug": "Slugs",
    "snail": "Snails",
    "wasp": "Wasps",
    "weevil": "Weevils",
}

if MODE == "proto":
    CLASSES = ["Bees", "Beetles", "Weevils"]

elif MODE == "full":
    CLASSES = ALL_CLASSES