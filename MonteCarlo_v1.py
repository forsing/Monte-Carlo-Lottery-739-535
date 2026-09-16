import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7_4686_k74.csv")
NEWEST_FIRST = False

# None = sva prethodna izvlačenja.
WINDOWS = (None, 50, 100, 250, 500, 1000)


def load_data(path):
    rows = []

    with path.open(encoding="utf-8-sig", newline="") as file:
        for line, fields in enumerate(csv.reader(file), 1):
            if not fields:
                continue

            try:
                row = tuple(int(value) for value in fields)
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: očekuju se celi brojevi."
                ) from error

            if (
                len(row) != 7
                or len(set(row)) != 7
                or any(value < 1 or value > 39 for value in row)
            ):
                raise ValueError(f"Neispravan red {line}: {row}")

            rows.append(row)

    if len(rows) < 100:
        raise ValueError("Potrebno je najmanje 100 redova.")

    if NEWEST_FIRST:
        rows.reverse()

    return rows


def prefix_counts(rows):
    """Kumulativne učestalosti za brzo računanje prozora."""
    prefix = [[0] * 40]

    for row in rows:
        counts = prefix[-1].copy()
        for number in row:
            counts[number] += 1
        prefix.append(counts)

    return prefix


def predict(prefix, end, window):
    """Koristi samo redove pre pozicije end."""
    if end < 1:
        raise ValueError("Nema prethodnih podataka.")

    start = 0 if window is None else max(0, end - window)
    counts = [
        prefix[end][number] - prefix[start][number]
        for number in range(40)
    ]

    # Pri istoj učestalosti prednost ima manji broj.
    ranking = sorted(
        range(1, 40),
        key=lambda number: (-counts[number], number),
    )
    return tuple(sorted(ranking[:7]))


def evaluate(rows, prefix, start, stop, window):
    """Predlog za svaki red pravi se pre uvida u taj red."""
    return [
        len(set(predict(prefix, index, window)) & set(rows[index]))
        for index in range(start, stop)
    ]


def model_name(window):
    return "Sva prethodna izvlačenja" if window is None else f"Poslednjih {window}"


def main():
    rows = load_data(CSV_PATH)
    prefix = prefix_counts(rows)
    n = len(rows)

    # 60% početna istorija, 20% izbor modela, 20% završni test.
    validation_start = int(n * 0.60)
    test_start = int(n * 0.80)

    validation = {
        window: evaluate(
            rows, prefix, validation_start, test_start, window
        )
        for window in WINDOWS
    }

    # Kod jednakog rezultata biramo model sa više istorije.
    best_window = max(
        WINDOWS,
        key=lambda window: (
            mean(validation[window]),
            n if window is None else window,
        ),
    )

    # Model se ovde više ne bira; ažuriraju se samo učestalosti
    # podacima koji prethode svakom narednom testiranom redu.
    test_hits = evaluate(
        rows, prefix, test_start, n, best_window
    )

    print(f"Učitano redova: {n}")
    print("\nProsečan broj pogodaka na validaciji:")
    for window in WINDOWS:
        print(f"{model_name(window)}: {mean(validation[window]):.4f}")

    average = mean(test_hits)
    margin = 1.96 * stdev(test_hits) / math.sqrt(len(test_hits))

    print(f"\nIzabran model: {model_name(best_window)}")
    print(f"Broj testiranih izvlačenja: {len(test_hits)}")
    print(f"Prosek na završnom testu: {average:.4f}")
    print(f"Teorijsko slučajno očekivanje: {49 / 39:.4f}")
    print(
        "Približan 95% interval za test prosek: "
        f"{average - margin:.4f}–{average + margin:.4f}"
    )

    print("\nPogodaka | Broj izvlačenja")
    distribution = Counter(test_hits)
    for hits in range(8):
        print(f"{hits:8d} | {distribution[hits]}")

    prediction = predict(prefix, n, best_window)

    print("\nPredlog za sledeće izvlačenje, CSV format:")
    print("b1,b2,b3,b4,b5,b6,b7")
    print(",".join(map(str, prediction)))


if __name__ == "__main__":
    main()



"""
Učitano redova: 4686

Prosečan broj pogodaka na validaciji:
Sva prethodna izvlačenja: 1.2327
Poslednjih 50: 1.2508
Poslednjih 100: 1.3020
Poslednjih 250: 1.2433
Poslednjih 500: 1.1996
Poslednjih 1000: 1.2625

Izabran model: Poslednjih 100
Broj testiranih izvlačenja: 938
Prosek na završnom testu: 1.2900
Teorijsko slučajno očekivanje: 1.2564
Približan 95% interval za test prosek: 1.2288–1.3511

Pogodaka | Broj izvlačenja
       0 | 209
       1 | 359
       2 | 270
       3 | 89
       4 | 11
       5 | 0
       6 | 0
       7 | 0

Predlog za sledeće izvlačenje, CSV format:
b1,b2,b3,b4,b5,b6,b7
1,7,14,16,18,28,31
"""
