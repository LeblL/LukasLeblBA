# Datasets

Dieser Ordner beschreibt die Datenstruktur des Projekts. Die grossen
Parquet-Dateien werden nicht als Codeartefakte versioniert und sind nicht Teil
einer schlanken Abgabe-ZIP.

## Datenstatus

Die Analyse wurde lokal mit DB-Haltdaten von November 2025 bis August 2026,
DWD-Wetterdaten und daraus erzeugten Zwischen- bzw. Modellierungsdaten
durchgefuehrt. Die Parquet-Dateien sind wegen ihrer Groesse in `.gitignore`
ausgeschlossen.

## Erwartete lokale Struktur

```text
datasets/
|-- monthly/       monatliche DB-Rohdaten, Nov. 2025 bis Aug. 2026
|-- metadata/      Stationskoordinaten und DB-DWD-Stationsmapping
|-- weather/       DWD-Stundenwerte fuer FF, FX, RR und TU
|-- processed/     erzeugte Zwischendatensaetze der Datenaufbereitung
`-- modeling/      finale Splits und Feature-Dateien fuer die Notebooks
```

## Reproduktion

Fuer einen vollstaendigen Neuaufbau muessen die Rohdaten lokal wieder in
`datasets/monthly/` bereitgestellt und anschliessend die Skripte aus
`scripts/data prep iteration 2/` sowie `scripts/modeling/` ausgefuehrt werden.

Die Modellierungsnotebooks setzen insbesondere die Dateien unter
`datasets/modeling/features/` voraus. Die deskriptive Analyse verwendet
zusaetzlich den Trainingssplit unter `datasets/modeling/splits/train.parquet`.
