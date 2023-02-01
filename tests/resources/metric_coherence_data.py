import pandas as pd
from py4ai.data.model.ml import PandasDataset


docs = PandasDataset(
    features=pd.Series(
        [
            "pippo è andato a pescare con topolino",
            "pippo e topolino sono andati a pescare",
            "pluto cane di topolino",
            "pippo porta in giro il cane pluto",
            "pluto cane topolino pippo",
            "pippo e pippotte",
            "pippotte non esiste",
        ]
    )
)

dictMapKeyDoc = {
    "pippo": {0, 1, 3, 4, 5},
    "pluto cane": {2, 4},
    "topolino": {0, 1, 2, 4},
    "cane pluto": {3},
    "pluto": {2, 3, 4},
    "pippotte": {5, 6},
    "non esiste": {6},
}

df = pd.DataFrame(
    [
        "pippo",
        "pluto cane",
        "topolino",
        "cane pluto",
        "pluto",
        "cane pluto",
        "pippotte",
        "non esiste",
    ]
)
topicDataset = PandasDataset(
    df, labels=pd.DataFrame(["topolino"] * 4 + ["pluto"] * 2 + ["pippotte"] * 2)
)
