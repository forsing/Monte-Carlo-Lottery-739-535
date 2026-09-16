import csv
from collections import Counter
from pathlib import Path
from statistics import mean

CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto5s_186_k73.csv")
NEWEST_FIRST = False
WINDOWS = (None, 50, 100, 250, 500, 1000)


def validate_row(row, line):
    if (
        len(row) != 6
        or len(set(row[:5])) != 5
        or any(not 1 <= x <= 35 for x in row[:5])
        or not 1 <= row[5] <= 10
    ):
        raise ValueError(
            f"Neispravan red {line}: {row}. "
            "Potrebno je 5 različitih brojeva 1–35 i dopunski 1–10."
        )


def load_data(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for line, fields in enumerate(csv.reader(file), 1):
            if not fields:
                continue
            try:
                row = tuple(int(x) for x in fields)
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: očekuju se celi brojevi."
                ) from error

            validate_row(row, line)
            rows.append(row)

    if len(rows) < 100:
        raise ValueError("Potrebno je najmanje 100 redova.")

    return rows[::-1] if NEWEST_FIRST else rows


def prefix_counts(draws, maximum):
    prefix = [[0] * (maximum + 1)]
    for draw in draws:
        counts = prefix[-1].copy()
        for number in draw:
            counts[number] += 1
        prefix.append(counts)
    return prefix


def predict(prefix, end, window, count):
    """Koristi samo podatke pre pozicije end."""
    if end < 1:
        raise ValueError("Nema prethodnih podataka.")

    start = 0 if window is None else max(0, end - window)
    ranking = sorted(
        range(1, len(prefix[0])),
        key=lambda x: (-(prefix[end][x] - prefix[start][x]), x),
    )
    return tuple(sorted(ranking[:count]))


def evaluate(draws, prefix, start, stop, window, count):
    return [
        len(set(predict(prefix, i, window, count)) & set(draws[i]))
        for i in range(start, stop)
    ]


def model_name(window):
    return "Sva prethodna" if window is None else f"Poslednjih {window}"


def analyze(draws, maximum, count, title):
    n = len(draws)
    prefix = prefix_counts(draws, maximum)
    validation_start, test_start = int(n * 0.60), int(n * 0.80)

    validation = {
        w: evaluate(draws, prefix, validation_start, test_start, w, count)
        for w in WINDOWS
    }

    best = max(
        WINDOWS,
        key=lambda w: (
            mean(validation[w]),
            n if w is None else min(w, n),
        ),
    )

    hits = evaluate(draws, prefix, test_start, n, best, count)

    print(f"\n{title}")
    for w in WINDOWS:
        print(f"  {model_name(w)}: validacija {mean(validation[w]):.4f}")

    print(f"Izabran model: {model_name(best)}")
    print(
        f"Test: {len(hits)} izvlačenja; "
        f"prosečno pogodaka: {mean(hits):.4f}"
    )
    print(f"Teorijsko slučajno očekivanje: {count * count / maximum:.4f}")

    print("Pogodaka | Broj izvlačenja")
    distribution = Counter(hits)
    for k in range(count + 1):
        print(f"{k:8d} | {distribution[k]}")

    return predict(prefix, n, best, count), hits


def main():
    rows = load_data(CSV_PATH)
    print(f"Učitano redova: {len(rows)}")

    main_numbers, main_hits = analyze(
        [row[:5] for row in rows],
        35,
        5,
        "Glavni brojevi 5/35",
    )

    bonus, bonus_hits = analyze(
        [(row[5],) for row in rows],
        10,
        1,
        "Dopunski broj 1/10",
    )

    full_hits = sum(
        a == 5 and b == 1
        for a, b in zip(main_hits, bonus_hits)
    )
    print(f"\nPogodaka 5+1 na testu: {full_hits}")

    print("\nPredlog, CSV format:")
    print("b1,b2,b3,b4,b5,dopunski")
    print(",".join(map(str, main_numbers + bonus)))


if __name__ == "__main__":
    main()



"""
Učitano redova: 186

Glavni brojevi 5/35
  Sva prethodna: validacija 0.6757
  Poslednjih 50: validacija 0.7838
  Poslednjih 100: validacija 0.5676
  Poslednjih 250: validacija 0.6757
  Poslednjih 500: validacija 0.6757
  Poslednjih 1000: validacija 0.6757
Izabran model: Poslednjih 50
Test: 38 izvlačenja; prosečno pogodaka: 0.7105
Teorijsko slučajno očekivanje: 0.7143
Pogodaka | Broj izvlačenja
       0 | 16
       1 | 17
       2 | 5
       3 | 0
       4 | 0
       5 | 0

Dopunski broj 1/10
  Sva prethodna: validacija 0.1081
  Poslednjih 50: validacija 0.1351
  Poslednjih 100: validacija 0.1351
  Poslednjih 250: validacija 0.1081
  Poslednjih 500: validacija 0.1081
  Poslednjih 1000: validacija 0.1081
Izabran model: Poslednjih 100
Test: 38 izvlačenja; prosečno pogodaka: 0.1053
Teorijsko slučajno očekivanje: 0.1000
Pogodaka | Broj izvlačenja
       0 | 34
       1 | 4

Pogodaka 5+1 na testu: 0

Predlog, CSV format:
b1,b2,b3,b4,b5,dopunski
6,8,32,34,35,5
"""
