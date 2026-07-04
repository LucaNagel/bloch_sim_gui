# Detaillierter Plan: 3D-Sequenzsimulation mit Pulseq-Unterstützung

Dieses Dokument hält den ursprünglichen Implementierungsplan dauerhaft fest.
Der aktuelle Umsetzungsstand, bekannte Einschränkungen und die nächste konkrete
Aufgabe stehen in [sequence_simulation_roadmap.md](sequence_simulation_roadmap.md).
Verbindliche technische Entscheidungen stehen in
[sequence_simulation_architecture.md](sequence_simulation_architecture.md).

## Zusammenfassung

Ziel ist eine physikalisch gekoppelte Sequenzsimulation für 3D-Objekte. Der
erste produktive Stand umfasst:

- RF sowie Gx/Gy/Gz,
- voxelweises T1, T2, Protonendichte, Chemical Shift und B0,
- ADC-Signal, finale Magnetisierung und optionale Checkpoints,
- Pulseq 1.5.0 über PyPulseq,
- speichereffiziente native Python-/C-Ausführung.

Bestehende APIs und GUI-Funktionen bleiben kompatibel. Browser/WASM,
Tx/Rx-Karten und mehrere chemische Spezies pro Voxel folgen später.

## Öffentliche APIs und Datenmodell

- Neues Paket `blochsimulator.sequence` mit unveränderlichen Ereignistypen:
  - `RFEvent`: Startzeit, komplexe RF-Werte in Hz, Raster, Frequenz- und
    Phasenoffset.
  - `GradientEvent`: Achse, Startzeit, Werte in Hz/m und Zeitraster.
  - `ADCEvent`: Startzeit, Samplezahl, Dwell-Time, Frequenz- und Phasenoffset.
  - `SequenceProgram`: geordnete Ereignisse, Gesamtdauer, Quelle, Version und
    Metadaten.
- Kanonische Einheiten:
  - Zeit: Sekunden,
  - Position: Meter,
  - RF: Hz,
  - Gradient: Hz/m,
  - Off-Resonanz: Hz.
  Der bestehende Gauss-/G/cm-Pfad wird nur im Legacy-Adapter umgerechnet.
- Neue Ladefunktion:

  ```python
  program = load_pulseq(path, strict=True)
  ```

  Sie unterstützt Pulseq bis einschließlich 1.5.0 und wirft für 1.5.1
  zunächst `UnsupportedPulseqVersionError`.
- Legacy-Adapter:

  ```python
  program = SequenceProgram.from_legacy(
      b1_gauss,
      gradients_g_per_cm,
      time_s,
      adc_times_s=(),
  )
  ```

  Ohne explizite ADC-Zeitpunkte werden nur Endzustand und Checkpoints
  berechnet.
- Neue Simulation:

  ```python
  result = BlochSimulator(...).simulate_sequence(
      program,
      phantom,
      checkpoints_s=(),
      chunk_voxels=None,
  )
  ```

- `SequenceSimulationResult` enthält:
  - `signal`: komplexes ADC-Signal,
  - `adc_times_s`,
  - `final_magnetization` mit Form `(*phantom.shape, 3)`,
  - optionale `checkpoint_magnetization`,
  - `checkpoint_times_s`,
  - Sequenz-, Objekt-, Einheiten- und Laufzeitmetadaten,
  - `to_dict()` und später `to_xarray()`.
- `Phantom` erhält `b0_map` und `chemical_shift_map`.
  - Effektiver Offset: `b0_map + chemical_shift_map`.
  - Das alte `df_map` bleibt lesbar, darf aber nicht gemeinsam mit den neuen
    Karten angegeben werden und wird als Legacy-B0-Karte übernommen.
  - Alte NPZ/HDF5-Dateien bleiben ladbar; neue Dateien speichern beide Karten
    getrennt.

## Implementierungsmeilensteine

### 0. Baseline und Korrektheitsfundament

- Den fehlerhaften absoluten Phantom-Import reparieren.
- Sequenz-, Gradienten- und Zeitlängen überall streng validieren.
- Voxelkoordinaten als echte Voxelzentren erzeugen:
  `-FOV/2 + Δx/2 … FOV/2 - Δx/2`.
- Gemeinsame, getestete Einheitenkonvertierungen einführen.
- Bekannte Gradientenkonvertierung im K-Space-Modul korrigieren.
- `additional_frequencies` und `use_grouped` entweder korrekt anbinden oder aus
  dem internen neuen Pfad entfernen; keine wirkungslosen Optionen übernehmen.
- Bestehende uncommittete Änderungen an README und Sweep-Dateien unangetastet
  lassen.
- Den damaligen vollständigen Testabbruch in `test_export_simple.py` als
  Baselineproblem dokumentieren; neue numerische Tests müssen unabhängig davon
  laufen.

Abnahme:

- Ein `2×2×2`-Phantom kann über den bestehenden Pfad simuliert werden.
- Bestehende gezielte RF-/Frequenz-/Inputtests bleiben grün.
- Einheiten- und Koordinatentests sind vorhanden.

### 1. Sequenz-IR und Compiler

- Ereignismodell und Legacy-Adapter implementieren.
- `SequenceCompiler` erstellt eine interne `CompiledSequence` mit:
  - variablen Zeitintervallen,
  - komplexem `rf_hz`,
  - `gradient_hz_per_m[:, 3]`,
  - ADC-Indizes und vorab berechneter komplexer Demodulation,
  - Checkpoint-Indizes.
- Zeitsemantik:
  - Magnetisierung wird über `[t_i, t_{i+1})` propagiert.
  - ADC und Checkpoints lesen den Zustand exakt am Intervallende.
  - Ein ADC-Sample bei `t=0` liest den Initialzustand.
- RF-aktive Bereiche werden auf RF-/Gradientenänderungen gerastert.
- RF-freie Bereiche werden zwischen ADC-, Checkpoint- und Blockgrenzen
  analytisch zusammengefasst:
  - Gradienten werden über ihre Fläche integriert,
  - Relaxation und z-Phasenrotation werden in einem Schritt berechnet,
  - lange Delays erzeugen keinen Mikrosekunden-Zeitvektor.
- Checkpoint-Zeiten außerhalb bestehender Grenzen werden als neue Grenzen
  eingefügt.
- Überlappende RF-/Gradientenereignisse werden addiert; überlappende Ereignisse
  derselben Hardwareachse werden als ungültige Sequenz gemeldet.

Abnahme:

- Kompilierte Gesamtdauer, RF-Fläche, Gradientenmomente und ADC-Zeiten stimmen
  mit den Eingaben überein.
- RF-freies Zusammenfassen liefert denselben Zustand wie feine Rasterung.
- Der Compiler allokiert nichts proportional zu
  `Sequenzdauer / kleinstem Raster`, wenn lange Delays vorliegen.

### 2. Streaming-C-Kern

- Bestehende C-Funktionen unverändert erhalten.
- Neue C-/Cython-Funktion in kanonischen Frequenzeinheiten ergänzen:
  - Eingaben: kompilierte Intervalle, aktive Voxel, T1/T2, Gesamt-Δf, PD und
    Initialmagnetisierung.
  - Zustandsvektor bleibt während des Laufs voxelweise im Speicher.
  - Ausgabe nur an ADCs, Checkpoints und am Ende.
- Signalberechnung im Bloch-Lauf:

  ```text
  S(t_n) = sum_v rho_v [Mx(v,t_n) + i My(v,t_n)] D_n
  ```

  mit vorab berechnetem Receiver-Demodulationsfaktor `D_n`.
- Gradientenphase wird ausschließlich im Bloch-Kern berechnet; keine zweite
  K-Space-Phasenmultiplikation.
- Voxel werden in speicherbudgetabhängigen Chunks verarbeitet.
- OpenMP verwendet threadlokale ADC-Signalpuffer, die nach jedem Chunk reduziert
  werden.
- Abbruch wird zwischen Chunks geprüft; dadurch kann die spätere GUI zuverlässig
  abbrechen.
- Speicherabschätzung berücksichtigt:
  - Voxelzustände,
  - threadlokale ADC-Puffer,
  - finale Magnetisierung,
  - explizit angeforderte Checkpoints.
- Hintergrundvoxeln werden nicht simuliert und anschließend auf die Objektform
  zurückprojiziert.

Abnahme:

- Peak-Speicher skaliert mit
  `Nvoxel + Nadc×Nthreads + Ncheckpoint×Nvoxel`, nicht mit `Ntime×Nvoxel`.
- Streaming- und bestehender Kern stimmen für kleine Referenzfälle numerisch
  überein.
- Ein synthetisches `64³`-Phantom mit 1.000 ADC-Samples und ohne Checkpoints
  bleibt unter dem berechneten Speicherbudget.
- Endpoint und ADC-Signal sind unabhängig von der Chunkgröße.

### 3. Python-API und Objektintegration

- `simulate_sequence()` an `BlochSimulator` anbinden.
- `Phantom`-Migration für B0 und Chemical Shift implementieren.
- PD wird ausschließlich als Signal-/Gleichgewichtsgewicht behandelt; die
  normierte Magnetisierung bleibt dimensionslos.
- Chemical Shift im ersten Stand: genau ein Offset pro Voxel.
- `received_signal` des alten Phantom-Pfads bleibt kompatibel, wird aber nicht
  als neue ADC-Simulation ausgegeben.
- Ergebnisexport um ADC-Zeitachse, finale Magnetisierung, Checkpoints und
  Sequenzmetadaten erweitern.
- Aussagekräftige Fehler für:
  - fehlende ADCs,
  - ungültige Kartenformen,
  - NaN/Inf,
  - T1/T2 ≤ 0 in aktiven Voxeln,
  - Checkpoints außerhalb der Sequenz,
  - überschrittenes Speicherbudget.

Abnahme:

- Python-End-to-End-Beispiele für FID, Spin-Echo und Gradient-Echo.
- B0- und Chemical-Shift-Karten erzeugen die erwartete Phasenentwicklung.
- Alte `simulate()`-Aufrufe liefern unveränderte Formen und Schlüssel.

### 4. Pulseq-1.5.0-Import

- Optionales Extra ergänzen:

  ```toml
  pulseq = ["pypulseq>=1.5.0,<1.5.1"]
  ```

- Nur öffentliche PyPulseq-APIs zum Lesen und Dekomprimieren verwenden.
- Pulseq-Blöcke einschließlich Event-Delay in `SequenceProgram` überführen.
- RF-, beliebige und trapezförmige Gradienten sowie ADC-Events unterstützen.
- ADC-Zeitpunkte aus PyPulseq übernehmen, nicht unabhängig neu herleiten.
- RF-/ADC-Frequenz und -Phase in absolute Sample-Demodulationsphasen
  übersetzen.
- Labels, Trigger und nichtphysikalische Extensions als Metadaten erhalten und
  mit Warnung ignorieren.
- Pulseq ≥1.5.1, Soft Delays und unbekannte Extensions explizit ablehnen.
- Import bleibt optional: Ohne PyPulseq funktionieren interne und
  Legacy-Sequenzen weiterhin.
- Offizielle Pulseq-Beispielsequenzen als Test-Fixtures verwenden; nur kleine,
  lizenzkompatible Dateien ins Repository aufnehmen.

Abnahme:

- Pulseq-FID, GRE und EPI werden geladen.
- Gesamtdauer, RF, Gradientenmomente und ADC-Zeiten stimmen mit PyPulseq überein.
- Nicht unterstützte Versionen und Extensions führen zu klaren,
  reproduzierbaren Fehlern.
- Pulseq-Datei → Simulation → ADC-Signal funktioniert ohne GUI.

### 5. Integrierter GUI-Sequenzbereich

- Neuen Tab „Sequence Simulation“ ergänzen; bestehende Phantom- und
  K-Space-Tabs bleiben zunächst erhalten.
- Bereiche:
  - interne Sequenz oder `.seq` auswählen,
  - Sequenzübersicht mit RF/Gx/Gy/Gz und ADC-Markern,
  - Pulseq-Version, Dauer, Block- und Samplezahlen,
  - Phantom und B0-/Chemical-Shift-Karten,
  - Checkpoint-Auswahl und Speicherprognose,
  - ADC-Signal, K-Space-Trajektorie und finale Magnetisierung.
- Simulation in Worker-Thread ausführen; Fortschritt und Abbruch zwischen
  Chunks.
- Keine vollständige zeitaufgelöste Magnetisierung als Standardoption anbieten.
- K-Space aus integriertem Gradientenmoment und ADC-Zeitpunkten ableiten.
- Vorhandene Phantom-Darstellung wiederverwenden, aber Signal nicht erneut
  analytisch phasenkodieren.
- PyInstaller-Paket enthält PyPulseq erst ab diesem GUI-Meilenstein; WASM
  blendet Pulseq-Import zunächst aus.

Abnahme:

- `.seq` laden, Phantom wählen, simulieren und Signal darstellen funktioniert
  ohne Wechsel zwischen getrennten Simulationspfaden.
- Abbruch beendet eine laufende Chunk-Simulation kontrolliert.
- Speicherwarnungen erscheinen vor der Allokation.

### 6. Nachgelagerte Erweiterungen

- Single-Tx: komplexe voxelweise B1+-Skalierung.
- Multi-Tx: RF pro Sendekanal und kohärente Summe der B1+-Felder.
- Multi-Rx: komplexe Sensitivitätskarte pro Empfangskanal und Signalform
  `(n_adc, n_coils)`.
- Mehrere unabhängige chemische Spezies pro Voxel.
- Pulseq 1.5.1 über aktualisiertes PyPulseq oder den offiziellen C++-Reader.
- WASM-Port mit identischer Python-API und single-threaded Chunk-Kern.
- Rekonstruktion bleibt ein separater Verarbeitungsschritt auf dem simulierten
  ADC-Signal.

## Testplan

- Analytische Physik:
  - freie T1-/T2-Relaxation,
  - 90°-/180°-Hardpulse,
  - konstante Off-Resonanz,
  - Gradientphase eines Punktobjekts,
  - B0 plus Chemical Shift,
  - ADC-Receiverphase und Frequenzdemodulation.
- Compiler:
  - lange Delays,
  - variable RF-/Gradientenraster,
  - trapezförmige und beliebige Gradienten,
  - ADC an Blockgrenzen,
  - Checkpoints vor, auf und zwischen Ereignisgrenzen.
- 3D:
  - uniformes `2³`-Referenzobjekt,
  - maskiertes Objekt,
  - räumlich variierendes T1/T2/PD/B0,
  - Chunkgrößen 1, kleine Chunks und automatisch gewählt.
- Pulseq:
  - FID, GRE und EPI,
  - RF-/Gradientenüberlagerung,
  - mehrere ADC-Blöcke,
  - Versions- und Extensionfehler.
- Regression:
  - bestehende Sequenzklassen und `simulate()` unverändert,
  - alte Phantomdateien ladbar,
  - Export und Xarray-Konvertierung.
- Numerische Standardtoleranzen:
  - analytische Rotation/Relaxation: `rtol=1e-8`, `atol=1e-10`,
  - Streaming gegen Legacy-Kern: `rtol=1e-8`, `atol=1e-9`,
  - Zeitpunkte: maximal halbes spezifiziertes Raster.
- CI:
  - neue numerische Tests headless ohne Qt,
  - GUI-Smoke-Tests unter Xvfb,
  - native Builds für Linux, macOS und Windows,
  - WASM erst ab dem späteren Port-Meilenstein.

## Dokumentation und Fortführung über Sessions

- `docs/sequence_simulation_architecture.md` ist die dauerhafte technische
  Referenz für:
  - Einheiten und Vorzeichen,
  - Zeit- und ADC-Semantik,
  - Datenmodelle,
  - Compilerregeln,
  - C-/Python-Datenfluss,
  - Speicher- und Parallelisierungsmodell.
- `docs/sequence_simulation_roadmap.md` wird nach jedem Arbeitsabschnitt
  aktualisiert mit:
  - Status jedes Meilensteins,
  - erledigten und offenen Punkten,
  - bekannten Fehlern,
  - zuletzt ausgeführten Tests,
  - konkreter nächster Aufgabe.
- Jede neue öffentliche Klasse und Funktion erhält Docstrings mit Formen,
  Einheiten und Fehlerfällen.
- Neue Physikentscheidungen werden im Architektur-Dokument unter
  „Design Decisions“ festgehalten.
- Beispiele werden erst aufgenommen, wenn ihr zugehöriger Meilenstein getestet
  ist.
- Bestehende Nutzeränderungen werden nicht überschrieben oder in sachfremde
  Änderungen einbezogen.

## Festgelegte Annahmen

- Erster Meilenstein: Kern und Python-API vor GUI.
- Plattform zunächst natives Python/Desktop; WASM später.
- Pulseq-Ziel zunächst 1.5.0 über optionales PyPulseq.
- B0 ist im ersten produktiven Stand enthalten; Tx/Rx später.
- Chemical Shift zunächst als ein Offset pro Voxel.
- Standardausgabe: ADC-Signal, finale Magnetisierung und explizite Checkpoints.
- Keine Exchange-, Diffusions-, Flow- oder Motion-Physik im ersten Stand.
- Bestehende öffentliche APIs bleiben rückwärtskompatibel.
