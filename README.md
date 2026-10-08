# LukasLeblBA

Python-Projekt fuer die Bachelorarbeit zur Prognose von SPNV-Haltverspaetungen
im Raum Koeln/Bonn. Die aktuelle Auswertung arbeitet mit DB-Haltdaten von
November 2025 bis August 2026, DWD-Wetterdaten und einem zeitlichen
Train/Validation/Test-Split.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Die Daten liegen lokal unter `datasets/`. Grosse Parquet-Dateien werden nicht als
Codeartefakte behandelt und koennen je nach Git-Status lokal bzw. ignoriert sein.

## Relevante Projektstruktur

```text
.
|-- datasets/
|   |-- monthly/       Rohdaten je Monat, Nov. 2025 bis Aug. 2026
|   |-- metadata/      Stationskoordinaten und DB-DWD-Stationsmapping
|   |-- weather/       DWD-Stundenwerte nach Wetterquelle
|   |-- processed/     Zwischenergebnisse der Datenaufbereitung
|   `-- modeling/      finale Splits und Feature-Dateien fuer die Modelle
|-- scripts/
|   |-- data prep iteration 2/  aktuelle Datenaufbereitung
|   |-- geo mapping/            Wetterstations-Zuordnung
|   `-- modeling/               Split- und Feature-Erstellung
|-- notebooks/
|   `-- modeling/      deskriptive Analyse, Modellauswahl, Interpretation,
|                      finale Evaluation
|-- outputs/           erzeugte Tabellen und Grafiken
|-- results/           vorgesehener Ablageort fuer Modellresultate
|-- requirements.txt   Python-Abhaengigkeiten
`-- README.md          diese Orientierung
```

## Wichtige Datenordner

- `datasets/monthly/`: monatliche Ausgangsdaten. Diese Dateien sind die Basis
  der SPNV-Datenaufbereitung.
- `datasets/metadata/`: manuelle bzw. vorbereitete Metadaten, insbesondere
  Koordinaten und das Mapping zwischen DB-Stationen und DWD-Wetterstationen.
- `datasets/weather/`: DWD-Wetterdaten. Die Unterordner `FF`, `FX`, `RR` und
  `TU` stehen fuer Wind, Windspitzen, Niederschlag sowie Temperatur/Feuchte.
- `datasets/processed/`: schrittweise erzeugte Parquet-Dateien der aktuellen
  Datenpipeline, z.B. Basisdatensatz, bereinigter Datensatz, Kalender- und
  Wetteranreicherung sowie der finale Modellierungsdatensatz.
- `datasets/modeling/splits/`: finaler zeitlicher Split:
  `train.parquet`, `validation.parquet`, `test.parquet`.
- `datasets/modeling/features/`: Feature-Dateien, die direkt in den
  Modellierungsnotebooks verwendet werden.

## Aktuelle Pipeline

Die aktuelle Datenpipeline ist `scripts/data prep iteration 2/`. Sie ersetzt den
frueheren Ansatz aus `scripts/data prep iteration 1/`.

Relevante Schritte:

1. `scripts/data prep iteration 2/06_build_spnv_base_dataset.py` baut den
   SPNV-Basisdatensatz aus den monatlichen Rohdaten.
2. `scripts/data prep iteration 2/14_build_clean_spnv_dataset.py` entfernt
   relevante Ausfaelle und setzt negative Verspaetungen fuer die Zielvariable
   auf `0`.
3. `scripts/data prep iteration 2/15_add_temporal_features.py` erzeugt
   Zeitmerkmale wie `planned_event_time`, `event_hour` und `event_weekday`.
4. `scripts/data prep iteration 2/16_add_calendar_features.py` ergaenzt
   NRW-Feiertage.
5. `scripts/data prep iteration 2/17_add_weather_features.py` verbindet die
   Haltdaten mit den DWD-Wetterdaten.
6. `scripts/data prep iteration 2/19_build_final_modeling_dataset.py` reduziert
   den angereicherten Datensatz auf die finalen Modellierungsspalten.
7. `scripts/modeling/01_create_temporal_split.py` erstellt Training,
   Validation und August-Testset.
8. `scripts/modeling/02_prepare_features.py` erstellt die finalen Feature-
   Parquetdateien fuer die Notebooks.

Die Skripte in `scripts/geo mapping/` sind relevant fuer die Herleitung und
Anpassung des Wetterstationsmappings. Sie werden nicht bei jedem
Pipeline-Durchlauf benoetigt, sind aber methodisch wichtig.

## Notebooks

- `notebooks/modeling/00_descriptive_analysis.ipynb`: deskriptive Analyse des
  Trainingsbestands und Plausibilitaetschecks.
- `notebooks/modeling/01_model_selection.ipynb`: Modellvergleich und Auswahl
  der finalen Modellkonfiguration.
- `notebooks/modeling/02_model_interpretation.ipynb`: Interpretation des
  ausgewaehlten Modells auf Basis des Juli-Validierungszeitraums.
- `notebooks/modeling/03_final_evaluation.ipynb`: finale Out-of-sample-
  Evaluation auf dem August-Testset.

## Erzeugte Outputs

- `outputs/descriptive_5_1/`: aktuelle Tabellen und Grafiken fuer die
  deskriptive Auswertung.
- `outputs/final_evaluation/`: finale Ist-Prognose-Grafiken fuer August.
- `outputs/model_interpretation/`: Grafiken zur Modellinterpretation.
- `outputs/descriptive_analysis/`: aeltere Exportvariante der deskriptiven
  Analyse. Fuer den aktuellen Stand ist `outputs/descriptive_5_1/` massgeblich.
- `results/modeling/`: Ablageort fuer Modellresultate; aktuell nur als
  Struktur vorgesehen.

## Nicht oder nur eingeschraenkt relevant

- `scripts/data prep iteration 1/`: historischer Ansatz aus Juni. Dieser
  Ansatz wurde verworfen und ist fuer den aktuellen Projektstand nicht
  massgeblich.
- `scripts/count_stations.py` und `scripts/parquet_file_stats.py`: kleine
  Hilfsskripte zur Exploration bzw. Dateipruefung, nicht Teil der finalen
  Pipeline.
- `.venv/`: lokale virtuelle Umgebung, nicht projektinhaltlich relevant.
- `.idea/`: lokale PyCharm-Projekteinstellungen.
- `__pycache__/`: automatisch erzeugte Python-Cache-Dateien.
- Aeltere Dateien direkt unter `outputs/`, z.B.
  `05_spnv_service_details.csv` und `05b_line_number_details.csv`, stammen aus
  Explorationsschritten und sind nicht Teil der finalen Ergebnisstruktur.

## Reproduktionshinweise

Die Modellierungsnotebooks setzen voraus, dass die Dateien unter
`datasets/modeling/features/` vorhanden sind. Wenn diese fehlen oder neu gebaut
werden sollen, zuerst die aktuelle Datenpipeline und danach die beiden Skripte
unter `scripts/modeling/` ausfuehren.

Der finale Testzeitraum ist August 2026. Der August wird in
`03_final_evaluation.ipynb` nur fuer die abschliessende Evaluation verwendet.
