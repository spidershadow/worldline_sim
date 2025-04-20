from pathlib import Path
import pandas as pd

DATA = {
    "uap_observations.csv": [
        ("year", "uap_count"),
        (2004, 27), (2010, 15), (2015, 91), (2020, 144), (2023, 366),
    ],
    "max_qubits.csv": [
        ("year", "max_qubits"),
        (1998, 2), (2000, 5), (2001, 7), (2011, 14),
        (2017, 72), (2019, 53), (2020, 65),
        (2021, 127), (2022, 433), (2023, 1121), (2024, 1581),
    ],
    "quantum_volume.csv": [
        ("year", "quantum_volume"),
        (2017, 4), (2018, 8), (2019, 16), (2020, 128),
        (2021, 512), (2022, 1024), (2023, 4096), (2024, 8192),
    ],
}


def main():  # pragma: no cover – utility script
    Path("data").mkdir(exist_ok=True)
    for fname, rows in DATA.items():
        header, *records = rows
        df = pd.DataFrame(records, columns=header)
        df.to_csv(Path("data") / fname, index=False)
    print("✅ CSVs written to ./data/")


if __name__ == "__main__":
    main() 