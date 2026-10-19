import os

from modules import scripts

localizations = {}


def list_localizations(dirname):
    localizations.clear()

    for file in os.listdir(dirname):
        fn, ext = os.path.splitext(file)
        if ext.lower() != ".json":
            continue

        localizations[fn] = [os.path.join(dirname, file)]

    for file in scripts.list_scripts("localizations", ".json"):
        fn, ext = os.path.splitext(file.filename)
        if fn not in localizations:
            localizations[fn] = []
        localizations[fn].append(file.path)

