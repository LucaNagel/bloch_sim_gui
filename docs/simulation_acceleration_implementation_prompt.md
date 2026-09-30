# Implementierungsauftrag: präzise Beschleunigung der Bloch-Simulation

Dieser Text ist als vollständiger Startprompt für eine neue Implementierungssession
gedacht. Projekt: `/Users/lucanagel/phd/bloch_simulator`. Planungsstand: 27.09.2026.
Relative Dateiverweise beziehen sich auf dieses Repository beziehungsweise auf den
Speicherort dieses Dokuments.

**Auftrag und Arbeitsweise**

Implementiere die folgenden Verbesserungen der Bloch-Simulation schrittweise.
Das Ziel sind deutlich kürzere Laufzeiten insbesondere bei Pyruvat-Laktat-Konversion,
B1-Feldkarten, vielen aktiven Voxeln, Subvoxel-Sampling und langen Sequenzen.
Numerische und physikalische Genauigkeit haben Vorrang vor Geschwindigkeit.
Dieser Auftrag verlangt Implementierung, Tests, Messungen und Dokumentation;
beende die Arbeit nicht nach einer weiteren Analyse oder einem Architekturentwurf.

Arbeite zunächst an den Float64-CPU-Pfaden. Entwickle darauf aufbauend einen
Float64-GPU-Prototyp und untersuche die bestehenden Metal-Verfahren weiter.
GPU-Code darf nur als auf Hardware validiert bezeichnet werden, wenn er tatsächlich
auf einer geeigneten GPU ausgeführt wurde. Fehlt diese Hardware, bearbeite die
unabhängigen CPU-, Planungs-, Validierungs- und Schnittstellenaufgaben weiter und
dokumentiere präzise, welche GPU-Prüfungen offen bleiben. Ein CPU-Fallback oder
ein gemockter Test gilt nicht als GPU-Beschleunigung.

Treffe normale Implementierungsentscheidungen eigenständig. Frage nur nach wirklich
fehlenden fachlichen Entscheidungen oder erforderlichem Hardwarezugang. Ändere
keine Modellannahmen oder wissenschaftlichen Fehlertoleranzen, um einen schnelleren
Kernel freigeben zu können. Nicht erfolgreich validierte Experimente bleiben
explizit experimentell und deaktivieren keinen korrekten CPU-Pfad.

Der Arbeitsbaum enthält bereits umfangreiche lokale Änderungen, unter anderem an
Sequenzgeneratoren, Spektral-Raum-Pulsen, Mausphantom und GUI. Prüfe zu Beginn
`git status` und die relevanten Diffs. Erhalte diese Änderungen einschließlich
ungetrackter Dateien. Kein Reset, Bereinigen oder Überschreiben fremder Arbeit.
Eine neue Arbeitskopie darf nicht unbemerkt den untersuchten lokalen Stand verlieren.
Keine Veröffentlichung, kein Push und kein Merge sind Teil dieses Auftrags.

Nutze portable Markdown-Syntax: Mathematik inline als `$...$`, abgesetzt als
`$$...$$`; keine MyST-Blöcke. Backticks sind für Code und Dateipfade vorgesehen,
nicht für mathematische Variablen. Verwende relative Dokumentlinks im Repository.

**Ausgangslage, die zuerst am aktuellen Code zu überprüfen ist**

Lies den [Analysebericht](simulation_acceleration_review.md), die
[CPU-Performancebeschreibung](dynamic_simulation_performance.md), den
[Metal-Machbarkeitsbericht](dynamic_metal_feasibility.md) und die
[Sequenzarchitektur](sequence_simulation_architecture.md).
Die [Kontrollmessungen](simulation_acceleration_benchmark.json) und das
[Reproduktionsskript](../benchmarks/benchmark_acceleration_review.py) dienen als
Startpunkt, nicht als universelle Leistungsprognose.

Die bekannten Befunde sind:

- Der gewöhnliche Sequenzsolver arbeitet bereits in C über vollständige Zeitverläufe
  pro Spin und verwendet Voxelblöcke. Seine `reference`- und `optimized`-Varianten
  liegen in derselben Erweiterung. Diese wird auf macOS/Linux bisher mit
  `-ffast-math` gebaut.
- Der dynamische Solver verwendet eine Python-Zeitschleife. `optimized` vermeidet
  viele Allokationen, ist aber kein vollständig parallelisierter Solver.
  `native_serial` und `native_parallel` ergänzen strikte Float64-Cython-Kernel.
- Schnelle dynamische RF-Blöcke sind derzeit auf bestimmte Zweipoolmodelle,
  statisches B0, Tx-Sensitivität exakt eins und höchstens 131.072 simulierte Spins
  begrenzt. Sie umfassen bis zu 512 RF-Intervalle. Der dynamische Solver expandiert
  derzeit das gesamte aktive Subvoxelobjekt gleichzeitig.
- Auch eine überall konstante Tx-Karte mit Faktor 0,9 deaktiviert diese RF-Blöcke.
  Räumliches B1 verwendet `_rf_rotate_spatial` mit wiederholten Arrayoperationen
  und Sinus-/Kosinusberechnungen.
- Exchange koppelt die Pools innerhalb eines Spins. Es gibt dafür keine räumliche
  Nachbarschaftskommunikation und keinen notwendigen Austausch zwischen Workern.
- Die bisherige M3-Metal-Probe überschritt die experimentelle Signal-NRMSE-Grenze
  von $10^{-3}$ bei langen bSSFP-Sequenzen. Zwei Float32-Zahlen für den Zustand
  behoben den Fehler nicht. Diese Grenze ist keine vom Nutzer bestätigte
  wissenschaftliche Toleranz.
- Das aktuelle Hybrid-Gate berücksichtigt Signalfehler; Endzustandsfehler werden
  zwar gemessen, aber noch nicht als vollständiges Freigabekriterium verwendet.

In der kleinen CPU-Messung mit 4.096 Spins betrug die Laufzeit mit Tx eins
266 ms für `optimized` und 95 ms für vier native Threads. Mit räumlichem Tx waren
es 250 ms beziehungsweise 222 ms. Acht Threads waren in diesem Beispiel langsamer
als vier. Die Zahlen beziehen sich auf unterschiedliche B1-Modelle und sind kein
isolierter kausaler Vergleich des B1-Rechenaufwands.

**Unveränderliche fachliche und numerische Anforderungen**

Erhalte die aktuelle Bedeutung von Ereignissen, ADC-Zeitpunkten, Demodulation,
Gradientenmomenten, Koordinaten, Frequenzreferenzen, Spulenempfindlichkeiten,
Signalgewichten, idealen Crushers, physikalischen Spoilern und Checkpoints.
Erhalte außerdem kinetischen Vorlauf, verzögerten Konversionsbeginn,
Zuflussverzögerungen, Konzentrationsverfolgung und Gleichgewichtspolarisation.

Das aktuelle dynamische Modell überträgt Magnetisierung durch irreversible
Konversion longitudinal. Ergänze weder transversalen kohärenten Austausch noch
Laktat-Rückkonversion. Zusätzliche bereits unterstützte Pools und
Vorläufer-Produkt-Konversionen müssen ihre bisherige Bedeutung behalten.
Übernimm nicht die Relaxation gegen den normierten Wert eins aus dem statischen
Solver in den hyperpolarisierten dynamischen Solver.

Übernimm anfangs unverändert die diskrete Fortschreibung: freie Halbzeitentwicklung,
RF-Schritt, freie Halbzeitentwicklung; ebenso die derzeitige Auswertung der
Zufluss- und Polarisationskurven an den Teilintervallgrenzen. Insbesondere darf
eine mathematisch anders integrierte zeitabhängige Quelle nicht als bloßes
Performance-Refactoring eingeführt werden.

Float64/Complex128 bleiben der Standard für Zustände, relevante Koeffizienten und
Signalakkumulation. Eine reduzierte Checkpoint-Speicherpräzision darf die weitere
Simulation nicht beeinflussen. Keine stillen Float32-Konvertierungen, keine
ungeprüfte Zeit-, Frequenz- oder B1-Quantisierung und keine ungeprüften Fast-Math-
oder FMA-Änderungen. Bitgleichheit und physikalische Genauigkeit sind getrennte
Eigenschaften und müssen auch getrennt berichtet werden.

**Codeübersicht für die Umsetzung**

| Bereich | Bestehende Dateien und zentrale Einstiegspunkte |
| --- | --- |
| Dynamische Physik und Treiber | [dynamic_phantom.py](../src/blochsimulator/dynamic_phantom.py): `simulate_dynamic_sequence`, `_free_step`, `_longitudinal_step`, `_prepare_longitudinal_step`, `_rf_rotate_spatial`, RF-Blockplanung, `observe` |
| Native dynamische Primitive | [dynamic_bloch_wrapper.pyx](../src/blochsimulator/dynamic_bloch_wrapper.pyx): RF-Rotation, gekoppelte longitudinale Schritte, persistente RF-Blöcke |
| Gewöhnlicher C-Solver | [bloch_core_modified.c](../src/blochsimulator/bloch_core_modified.c), [bloch_wrapper.pyx](../src/blochsimulator/bloch_wrapper.pyx), [Build-Konfiguration](../setup.py) |
| Dispatch und öffentliche API | [simulator.py](../src/blochsimulator/simulator.py): Kernelwahl, dynamische und statische Sequenzsimulation |
| Plan, Ergebnisse, Sampling | [compiler.py](../src/blochsimulator/sequence/compiler.py), [result.py](../src/blochsimulator/sequence/result.py), [spin_sampling.py](../src/blochsimulator/sequence/spin_sampling.py), [memory.py](../src/blochsimulator/memory.py) |
| Metal | [dynamic_metal_backend.py](../src/blochsimulator/dynamic_metal_backend.py), [Metal-Shader](../src/blochsimulator/metal/dynamic_bloch_precision.metal), [Objective-C++-Bridge](../src/blochsimulator/metal/dynamic_metal_probe.mm), [Cython-Bridge](../src/blochsimulator/dynamic_metal_probe_wrapper.pyx) |
| GUI und Einstellungen | [dialogs.py](../src/blochsimulator/ui/dialogs.py), [sequence_simulation_widget.py](../src/blochsimulator/ui/sequence_simulation_widget.py) |
| Bestehende Leistungsmessungen | [benchmark_kernels.py](../benchmarks/benchmark_kernels.py), [benchmark_sequences.py](../benchmarks/benchmark_sequences.py), [Benchmark-Anleitung](../benchmarks/README.rst) |
| Genauigkeitsauswertung | [validate_dynamic_precision.py](../scripts/validate_dynamic_precision.py), [convergence.py](../src/blochsimulator/sequence/convergence.py), [convergence_suite.py](../src/blochsimulator/sequence/convergence_suite.py) |

Neue Module sind sinnvoll, wenn sie den großen dynamischen Treiber aufteilen.
Mögliche Verantwortlichkeiten sind Ausführungsplan, Speicherplanung, Reduktion,
CPU-Ausführung und GPU-Ausführung. Namen und Datenklassen sind Vorschläge; vermeide
parallele, voneinander abweichende Implementierungen derselben Physik.

**Phase 0: reproduzierbare Ausgangsbasis und Messinstrumentierung**

1. Prüfe Repository-Anweisungen, Interpreter, importierte Modulpfade,
   Erweiterungsstand, NumPy-/BLAS-Backend, OpenMP-Verfügbarkeit und vorhandene
   GPU-Hardware. Im Analysezeitpunkt funktionierte
   `/opt/miniconda3/envs/bloch-simulator/bin/python`; der allgemeine
   `/opt/miniconda3/bin/python` scheiterte an einem SciPy-Import.
   Prüfe diese Umgebung erneut, statt globale Pakete vorsorglich zu verändern.
2. Prüfe, dass die tatsächlich importierten Cython-Erweiterungen zum Quellstand
   passen. Ein vorhandenes `.so` beweist weder einen aktuellen Build noch eine
   funktionierende OpenMP-Parallelisierung. Falls ein nativer Build durch fehlende
   Werkzeuge oder eine nicht akzeptierte Xcode-Lizenz blockiert ist, dokumentiere
   dies und bearbeite unabhängige Aufgaben weiter; akzeptiere keine Lizenz selbst.
3. Sichere eine reproduzierbare Baseline für Unity-Tx, konstantes komplexes Tx,
   räumliches Tx, lange RF-Züge, ADC-lastige Sequenzen und größere Subvoxelgitter.
   Verwende synthetische Fixtures, damit lokal ignorierte Phantom- oder Sequenzdateien
   keine unbemerkte Voraussetzung der Tests sind. Vorhandene reale Projekte sind
   zusätzliche End-to-End-Fälle.
4. Messe getrennt Kompilierung, Koeffizienten-/Planerstellung, native Fortschreibung,
   ADC-Reduktion und Ergebnisaufbereitung. Nutze feine Instrumentierung nur im
   Profilmodus; sie darf normale Simulationsläufe nicht pro Intervall verteuern.
5. Protokolliere angeforderten und tatsächlichen Kernel, Fallbacks, effektive
   Threadzahl, Spin-/Intervall-/ADC-Zahlen, RF-Blockabdeckung, Cache-Hits/-Misses,
   Cache-Bytes, Speichermaximum und numerischen Vertrag. Erfasse später zusätzlich
   GPU-Gerät, FP64-Unterstützung, Transfermengen und Synchronisationszeiten.

Abnahme: reproduzierbare Baselines einschließlich Genauigkeitsmetriken liegen vor.
Zeitmessungen verwenden Aufwärmläufe und mindestens drei unabhängige Wiederholungen;
bei kleinen Unterschieden mehr Wiederholungen und Streuung ausweisen. Keine
Leistungsbehauptung allein aus einer höheren CPU-Auslastung ableiten.

**Phase 1: belastbare Referenzen und gemeinsame Genauigkeitsprüfung**

Erhalte die bestehenden dynamischen Referenzpfade und ihre bitweisen Regressionen.
Ergänze für den gewöhnlichen C-Solver einen strikten Referenzbuild ohne Fast Math
und ohne unkontrollierte Kontraktion. Nutze eine eindeutig erkennbare Buildvariante
oder ein getrenntes Erweiterungsziel. Verhindere, dass gemeinsam erzeugte
Objektdateien oder ein alter Modulimport die tatsächlichen Compilerflags verschleiern.
Die bisherige Auswahl `reference` allein ist noch kein Beleg für einen strikten Build.

Definiere maschinenlesbare numerische Verträge:

- `bitwise`: unveränderte Koeffizienten, Rechenfolge und Reduktion; entsprechende
  bestehende Tests bleiben exakt.
- `float64-close`: neue Fusion, Chunking, feste Reduktionsbäume, Propagatoren oder
  GPU-Rechenfolge; Abweichungen werden eng begrenzt und vollständig ausgewiesen.
- `experimental-mixed`: separate explizite Auswahl; kein Ersatz für den
  Float64-Produktionsvertrag und keine automatische Toleranzlockerung.

Erweitere die gemeinsame Vergleichsauswertung für Gesamt- und Poolsignale,
spulenaufgelöste Signale, finalen Zustand, Checkpoints, Phasen und gegebenenfalls
K-Raum/Rekonstruktionen. Vergleiche Pools einzeln, damit starkes Pyruvat schwaches
Laktat nicht verdeckt. Berichte relative L2-Fehler, maximale absolute Fehler,
gegen den Referenzpeak normierte Fehler und Fehlerwachstum über die Messdauer.
Für Nullreferenzen und schwache Signale werden physikalisch begründete absolute
Skalen benötigt; ein winziger Nennerersatz ist kein sinnvolles Abnahmekriterium.

Bewahre vorhandene Testtoleranzen. Für kleine native RF-Fusionsfälle existieren
beispielsweise bereits `rtol=2e-13` und `atol=2e-13`. Verwende diese nicht blind für
beliebig skalierte Signale. Definiere vor der jeweiligen Optimierung dokumentierte
Grenzen pro Testklasse und physikalischer Skala. Orientiere Float64-Grenzen an
strikter Referenz und bestehendem Float64-Fehlerniveau; erhöhe sie nicht nachträglich,
damit ein neuer Kernel besteht. Die experimentelle Metal-Grenze $10^{-3}$ ist für
die Float64-CPU-Abnahme ungeeignet.

Abnahme: jede weitere Phase kann denselben Vergleichsrunner verwenden. Strikter
und bisheriger C-Build sind unterscheidbar; Referenzabweichungen und
Diskretisierungsfehler werden getrennt behandelt.

**Phase 2: konstantes und räumliches B1 nativ unterstützen**

Beginne mit einer expliziten Tx-Klassifikation: überall eins, überall derselbe
endliche komplexe Wert, räumlich variierende Karte. Verwende exakte Gleichheit für
die konstante Klasse. Behandle Nullamplitude und komplexe Phase korrekt; die
Klassifikation darf keine nur ungefähr konstante Karte zusammenfassen.

Für konstantes Tx kann die effektive RF-Wellenform einmal vorbereitet werden.
Behalte den Unity-Pfad unverändert. Achte darauf, dass die bisherige räumliche
RF-Rotation und die skalare Rodrigues-Implementierung unterschiedliche
Gleitkomma-Rechenfolgen haben: mathematische Gleichheit rechtfertigt noch kein
`bitwise`-Versprechen. Falls erforderlich, übernehme die bisherige Ebenenbasis
in den neuen nativen Pfad und deklariere verbleibende Unterschiede ausdrücklich.

Ergänze danach ein natives räumliches RF-Primitiv. Bereite ortsabhängige Achsen und
Sinus-/Kosinuswerte in Float64 vor. Die Anwendung soll unmittelbar auf dem
transversalen Zustand und der longitudinalen Magnetisierung arbeiten und unnötige
Kopien zwischen komplexer und dreikomponentiger Darstellung vermeiden.
Integriere dieses Primitiv auch in persistente RF-Blöcke.

Cache wiederkehrende Kombinationen aus RF-Wert, exakter Zeitdauer und Tx-Karte
beziehungsweise Voxelblock. Begrenze den Cache nach Bytes. Vermeide eine Tabelle
aller Spins mal aller RF-Intervalle. Ein Cache über mehrere Simulationen muss
Änderungen an Feldkarten erkennen; ein Array-Objektbezeichner allein reicht dafür
nicht. Ein simulationslokaler Cache ist eine einfache sichere Ausgangslösung.

Tests: Tx eins, Null, 0,9, reiner Phasenfaktor, kombinierte Amplitude/Phase,
heterogene Karten, nichtkontiguierliche Eingaben, verschiedene Poolzahl,
Subvoxel-Sampling und bestehende longitudinale Treiber. Prüfe nicht nur Ergebnisse,
sondern dass der beabsichtigte native RF-Pfad tatsächlich ausgeführt wird.
Passe bisherige Tests, die genau diesen B1-Fallback erwarten, gezielt an;
unabhängige Fallback-Schutztests müssen bestehen bleiben.

Abnahme: konstantes Nicht-Unity-Tx verhindert RF-Fusion nicht mehr. Räumliches Tx
verwendet im unterstützten Modell native RF-Schritte und Blöcke. Die bisherigen
Referenzen bleiben verfügbar, und Genauigkeit sowie Zeitgewinn sind gemessen.

**Phase 3: gemeinsamer Ausführungsplan für CPU und GPU**

Trenne Sequenzkompilierung, Modellvorbereitung und Ausführung. Entwickle einen
internen Plan mit zusammenhängenden Arrays und expliziten Segmenten für RF,
RF-freie Entwicklung, ADC-Readout und Zustandsbeobachtung. Derselbe Plan soll später
von CPU und GPU konsumiert werden können.

Der Plan enthält mindestens die ursprünglichen Intervallgrenzen und -dauern,
RF-Werte, Gradientendaten, benötigte integrierte B0-Beiträge, Konversionsaktivität,
Quelleninformationen sowie ADC-, Crusher- und Checkpoint-Zuordnungen.
Sichere außerdem die bisherige Reihenfolge mehrerer Aktionen am selben Zeitpunkt.
Ein Segment darf intern mehrere Ereignisse enthalten, solange deren Reihenfolge
und Beobachtungen exakt erhalten bleiben.

Der neue Plan darf keine zusätzlichen Zeitrundungen einführen und nicht durch
Zusammenlegen von nahezu gleichen Intervallen die Diskretisierung ändern.
Identische Dauern dürfen gemeinsame Koeffizienten verwenden. Prüfe Cache-Schlüssel
auf Reproduzierbarkeit, aber quantisiere sie nicht stillschweigend.

Trenne globale Sequenzinformationen von voxelabhängigen Informationen. Erzeuge
ortsabhängige Koeffizienten und Zuflussdaten nur für den aktuellen Voxel-/Zeitblock.
Wiederhole den Compiler nicht pro Voxelblock. Die neue Planung soll auch bei vielen
einzigartigen Gradienten- oder RF-Werten speicherbegrenzt bleiben.

Abnahme: Planausführung über einen einfachen Vergleichsadapter reproduziert alle
bisherigen Beobachtungszeitpunkte und Ergebnisse. Ein gezielter Test mit RF,
ADC, Crusher, Quellenwechsel und Checkpoint an gemeinsamen Grenzen deckt die
Ereignisreihenfolge ab.

**Phase 4: vollständige native dynamische Fortschreibung**

Nutze den Plan für einen kompilierten Segmenttreiber. Ein Worker bearbeitet einen
Spin mit allen gekoppelten Pools und Konzentrationen. Innerhalb eines Segments
liegen Zeitfortschreibung, RF, Relaxation, Konversion, Quellen und ADC-Verarbeitung
im nativen Code. Vermittle nicht weiter jeden elementaren Zeitschritt über Python.

Baue die Unterstützung in dieser Reihenfolge aus: Zweipoolmodell ohne Zufluss;
konzentrationsverfolgter polarisierter Zufluss; konstantes und räumliches Tx;
dynamisches B0; verzögerte Konversion und räumliche Zuflussankunft;
übrige bereits vorhandene Zufluss-/Konzentrationskombinationen; zusätzliche Pools
einschließlich vorhandener Vorläufer-Produkt-Konversionen.

Verwende zunächst die bereits in Float64 vorbereiteten Koeffizienten und bewährten
analytischen Teilschritte. Generische Mehrpoolunterstützung darf die spezialisierte
Zweipoolschleife nicht unnötig verteuern. Bei zusätzlichen Produkten berücksichtige
die gesamte Vorläuferverlustrate in allen passenden Koeffizienten und deren
Cache-Schlüsseln; vermeide wiederholte zustandsunabhängige Exponentialauswertungen.

Fusioniere auch RF-freie Readouts mit ADC-Akkumulation. Führe Empfangssensitivitäten,
Subvoxelgewichte, Volumengewichte, Pools und Empfangsdemodulation in derselben
Konvention wie bisher fort. Die Signalreduktion ist eine eigene, in Phase 5
festgelegte Aufgabe und darf keine unkontrollierten Summationsrennen enthalten.

Erhalte einen expliziten CPU-Vergleichs-/Fallbackpfad für noch nicht abgedeckte
Modellkombinationen. Eine Teilfallback-Kette muss Zustände, Rechenzeit und Gründe
korrekt ausweisen. Unterstützte Felder dürfen niemals einfach ignoriert werden.
Ein Fall gilt erst als beschleunigt, wenn der native Pfad tatsächlich benutzt wird.

Abnahme: auch lange ADC-lastige Abschnitte laufen nativ. Die alten RF-Blockgrenzen
werden nicht lediglich vergrößert, während der eigentliche Intervalltreiber in
Python verbleibt. Das Profil zeigt die veränderte Verteilung der Laufzeit.

**Phase 5: Voxelblöcke, begrenzter Speicher und deterministische Reduktion**

Führe dynamisches Chunking nach Elternvoxeln ein. Alle Subspins eines Elternvoxels
gehören logisch zusammen. Expandiere jeweils nur den aktuellen Elternvoxelblock,
führe dessen Zustände durch die gesamte Sequenz und aggregiere Ergebnisse und
Checkpoints mit den ursprünglichen Gewichten. Das ist insbesondere für Fälle
oberhalb von 131.072 Gesamtspins erforderlich; das bisherige Limit darf nicht nur
entfernt werden, während die Speicherkomplexität unverändert bleibt.

Unterscheide Speicher-Chunks von festen Reduktionsgruppen. Lege stabile globale
Spin-IDs und einen festgelegten Summationsbaum fest, dessen Blätter nicht von der
OpenMP-Threadzahl oder frei gewählten Chunk-Größe abhängen. Nutze geordnete
partielle Summen oder einen Streaming-Baum. Summiere Teilergebnisse nicht in der
zufälligen Fertigstellungsreihenfolge von Workern.

Das neue Verfahren muss bei gleichem Backend und gleicher Rechenumgebung über
Wiederholung, Workerzahl und erlaubte Speicher-Chunk-Größen reproduzierbar sein.
Die neue feste Reduktion muss nicht bitgleich zur bisherigen NumPy-Summe sein;
ihre Abweichung gehört zum ausgewiesenen `float64-close`-Vertrag. Untersuche
paarweise oder kompensierte Float64-Summation an stark auslöschenden Signalen.
Löse die Reproduzierbarkeit nicht durch ein vollständiges Array aller
Spinbeiträge für alle ADC-Zeitpunkte.

Der Speicherplan berücksichtigt Eingaben, aktuelle Zustände, alle Pools,
Konzentrationen, Feldkarten, Cache-Tabellen, Empfangsspulen, native Kopien,
Reduktionspuffer, Checkpoints, finale Ergebnisse und später Gerätepuffer.
Das Arbeitsvolumen soll mit der gewählten Blockgröße wachsen; globale
Zeitachsen und angeforderte Ausgaben bleiben als unvermeidbare Größen separat
ausgewiesen. Kein verstecktes vollständiges Subvoxelarray vor der Speicherprüfung.

Erhalte Abbruch und Vorschau durch begrenzte native Aufrufe. Ein Elternvoxelblock
kann über mehrere Zeitsegmente hinweg resident bleiben; es braucht keinen
Python-Aufruf pro Spin und Zeitschritt. Zwischen Speicherblöcken zeigt eine
Vorschau nur den bisher aufsummierten Objektanteil und muss als solche erkennbar
sein. Fortschritt basiert auf abgearbeiteter Arbeit, nicht fälschlich nur auf
der Zeitposition des gerade bearbeiteten Blocks.

Tests: sehr kleine Budgets, unterschiedliche Chunk-Größen, Grenzfälle direkt
unter und über der früheren Spin-Grenze, Mehrspulenempfang, viele ADCs,
Checkpoints, starke Signalaufhebung, Abbruch mitten in einer Simulation und
vollständige Subvoxelgewichte. Ein winziges Budget führt vor der großen
Allokation zu einer verständlichen Fehlermeldung.

Abnahme: größere Objekte verlieren die native Ausführung nicht allein wegen der
Gesamtspinzahl. Gemessener Speicherverbrauch entspricht dem Plan mit dokumentierter
Reserve. Thread- und Chunk-Tests bestehen; Vorschau und Endergebnis werden nicht
miteinander verwechselt.

**Phase 6: Koeffizienten, Datenlayout und CPU-Skalierung optimieren**

Verwende die gemessenen Profile, um verbliebene Kosten zu priorisieren. Cache
Relaxations- und Konversionskoeffizienten nach exakten Dauern und Parametergruppen,
Phasenfaktoren nach den tatsächlich benötigten Feld-/Gradientenintegralen und
RF-Koeffizienten nach dem relevanten B1-/Wellenformzustand. Begrenze jeden Cache
und rechne seine Größe in das Speicherbudget ein. Prüfe Evictions und Nutzen
auch bei Sequenzen ohne wiederkehrende Motive.

Wähle das native Datenlayout anhand von zusammenhängenden Speicherzugriffen und
SIMD-Eignung. Vermeide doppelte vollständige Zustandsdarstellungen im neuen
Backend. Behalte notwendige Adapter zum öffentlichen Ergebnisformat an den
Ausgabegrenzen bei. SIMD und Fusion dürfen nicht durch Fast Math erkauft werden.

Prüfe OpenMP-Granularität und die kleinste sinnvolle Arbeitsmenge. Explizites
`native_serial` verwendet genau einen Worker. Bei automatischer Auswahl sollen
kleine Fälle keine unnötigen Parallelregionen starten. Verhindere verschachtelte
BLAS-/OpenMP-Überbelegung. Automatische Entscheidungen sollen aus dokumentierten
Arbeitsgrößen folgen; teure Kalibrierläufe bei jedem Simulationsstart vermeiden.

Prüfe zusätzlich den gewöhnlichen C-Sequenzsolver: Relaxationstabellen nach exakten
Dauern wiederverwenden, Gruppen-/Tabellenspeicher budgetieren und die vorhandenen
Voxelblöcke beibehalten. Beim älteren vollständigen Verlaufsmodus den tatsächlichen
Speicherbedarf früh prüfen; angeforderte Zeitverläufe nicht heimlich reduzieren.
Keine Spins allein aufgrund geringer momentaner Magnetisierung oder einer
vermeintlichen Schichtselektion verwerfen.

Abnahme: Leistung und Speicher sind gegen die schnellste korrekte bisherige
CPU-Konfiguration verglichen, nicht nur gegen den langsamen Python-Referenzpfad.
Kleine Fälle und Fälle mit schlechten Cache-Trefferraten sind im Vergleich enthalten.

**Phase 7: wiederkehrende Blöcke als affine Propagatoren untersuchen**

Entwickle einen separaten, explizit auswählbaren Prototyp für wiederkehrende
RF-/Entwicklungsblöcke. Das aktuelle lineare Modell erlaubt für einen geeigneten
Zustandsvektor eine Darstellung

$$
\mathbf x_{n+1}=A_n\mathbf x_n+\mathbf b_n.
$$

Bei zwei Pools mit Konzentrationsverfolgung umfasst der Zustand sechs
Magnetisierungskomponenten und zwei Konzentrationen. Verwende passende
mehrdimensionale oder homogen erweiterte Operatoren; ein einziges
$3\times3$-RF-Rotationsprodukt beschreibt die gekoppelte Dynamik nicht.

Komponiere zunächst exakt dieselben diskreten Teilschritte wie der Referenzsolver,
keine gröbere Ersatzintegration. Quellenanteile können sich trotz wiederkehrender
RF-Wellenform mit der absoluten Zeit ändern. Trenne deshalb gegebenenfalls den
wiederverwendbaren homogenen Anteil vom zeitabhängigen affinen Anteil.
Identische RF-Form allein ist kein ausreichender Cache-Schlüssel.

ADC und Checkpoints innerhalb eines Blocks erfordern passende Teilpropagatoren
oder eine Segmentierung. Crusher, Konversionswechsel und Quellen-/B0-Grenzen
müssen in der richtigen Reihenfolge berücksichtigt werden. Definiere einen
vollständigen Parameterschlüssel und einen begrenzten Cache pro Voxelblock.

Miss Vorbereitungszeit, Wiederverwendung, Speicher und Rundungsfehler. Aktiviere
diesen Pfad nur, wenn er einschließlich Vorbereitung schneller als der neue
native Treiber ist und dessen Genauigkeitskriterien einhält. Keine automatische
Nutzung allein aufgrund einer wiederholten RF-Pulsform.

Abnahme: entweder validierter Gewinn auf einem dokumentierten wiederholten
Sequenzmotiv oder ein begründetes experimentelles Ergebnis ohne Aktivierung.
Ein negativer Befund darf nicht als erfolgreiche Produktionsoptimierung gelten.

**Phase 8: Float64-GPU-Prototyp auf geeigneter Hardware**

Prüfe verfügbare, für diesen Auftrag zugängliche Hardware und deren native
Float64-Unterstützung. Wähle das Backend anhand dieser Hardware; bei NVIDIA ist
CUDA ein möglicher Kandidat. Prüfe aktuelle offizielle Dokumentation bei konkreten
API-/Buildentscheidungen. Nutze GPU-Abhängigkeiten optional und lazy geladen;
CPU-Installation und Import müssen ohne GPU-Pakete weiterhin funktionieren.

Übernimm den gemeinsamen Plan und beginne mit dem validierten Zweipoolmodell.
Ein GPU-Thread beziehungsweise eine geeignete feste Threadgruppe bearbeitet die
lokal gekoppelten Zustände. Verarbeite viele Intervalle pro Launch. Halte Zustände
und unveränderte Eingaben über Segmente auf dem Gerät. Übertrage primär reduzierte
ADC-Ergebnisse, angeforderte Checkpoints und Endzustände.

Verwende zunächst CPU-vorbereitete Float64-Koeffizienten, wo dies speicherverträglich
ist. Werden Exponential- oder trigonometrische Funktionen auf der GPU berechnet,
validiere deren Beitrag gesondert gegen die strikte CPU-Auswertung. Es darf
keine unbemerkte Nutzung von Float32-Intrinsics geben. Passe Compilerflags an den
ausgewiesenen numerischen Vertrag an.

Implementiere eine deterministische GPU-Reduktion mit festen Gruppen und
Float64-Akkumulation; keine reihenfolgeabhängigen globalen Atomic-Summen als
Standard. Vermeide die Ausgabe aller Spins mal aller ADC-Zeitpunkte.
Tile Readout-Ausgaben beziehungsweise Reduktionspuffer bei Bedarf, während die
Magnetisierung auf dem Gerät erhalten bleibt.

Miss kalte und warme Läufe getrennt. Berücksichtige Planaufbereitung,
Kompilierung, Transfers, Synchronisation, Reduktion und Ergebnisaufbereitung.
Vergleiche mit der schnellsten validierten CPU-Konfiguration derselben Physik
und Diskretisierung. Ein kurzer GPU-Kernel bei langsamerem Gesamtlauf ist kein
Erfolg. Prüfe anschließend B1, dynamisches B0, Zufluss, Subvoxel und Mehrspulen
schrittweise; unterstützte Kombinationen müssen explizit erkennbar sein.

Abnahme: tatsächliche GPU-Ausführung und numerischer Vertrag sind nachgewiesen,
End-to-End-Gewinn und Speicherbedarf gemessen. Fehlt geeignete Hardware, liefere
einen klar beschriebenen Hardware-Abnahmelauf und markiere die Phase als nicht
auf Hardware validiert. Keine neue Standardauswahl für einen ungeprüften Backendpfad.

**Phase 9: Metal-Präzision und Hybrid-Gate gezielt weiterentwickeln**

Behandle Metal unabhängig von einem nativen Float64-GPU-Backend. Eine veränderte
Zustandsspeicherung allein genügte in den bisherigen Tests nicht. Untersuche
isoliert die Genauigkeit von Orts-/Gradientenprodukten, B0 und Pooloffsets,
Phasenakkumulation, Argumentreduktion, Sinus/Kosinus und kinetischen Faktoren.

Prüfe ein High-/Low-Float32-Verfahren für kritische Argumente und Koeffizienten
sowie stabile Auswertungen kleiner Exponentialdifferenzen und fast gleicher
Zerfallsraten. Validierte Serien oder entsprechend genaue Funktionen können
dafür erforderlich sein. Miss Präzisionsgewinn und Kosten einzeln; rechne nicht
mit einer kostenlosen Float64-Emulation.

Erweitere das Hybrid-Gate um alle vom Ergebnis versprochenen Größen:
Gesamt-/Poolsignale, Endzustände, signifikante Phasen und, falls unterstützt,
Checkpoints und Rekonstruktionen. Unterscheide geschätzte Stichprobenfehler von
tatsächlichen Fehlern gegen eine vollständige Float64-Referenz. Ein bestandener
Stichprobentest beweist keine allgemeine Fehlergrenze für unbekannte Objekte.

Kalibrierungs- und Validierungsgruppen bleiben disjunkt. Prüfe starke
Subvoxel-Auslöschung, schwaches Laktat, heterogene B1-/B0-Felder und zeitliches
Fehlerwachstum. Eine Änderung der Korrektur darf nicht aus den zurückgehaltenen
Validierungsdaten lernen. CPU-Fallback muss unveränderte korrekte Ergebnisse
liefern und seine gesamte zusätzliche Rechenzeit ausweisen.

Prüfe zunächst einen kurzen Prefix, danach ein Volumen, mehrere Volumina und die
vollständige lange Sequenz. Falls bereits der kurze Versuch scheitert, analysiere
den Fehler vor weiteren teuren Läufen. Ein erfolgreiches Beispiel allein erlaubt
keine allgemeine Produktivfreigabe. Eine wissenschaftliche Toleranzlockerung ist
nicht Teil dieses Auftrags.

Abnahme: dokumentierte Fehlerursachen und bestandene vollständige Validierung
oder ein begründeter negativer Befund. Bis dahin bleiben Metal und Hybrid
experimentell, mit korrektem Fallback und ohne neue Genauigkeitsversprechen.

**Phase 10: API, GUI, Export und Freigabe**

Erhalte bisherige Kernelbezeichnungen und Aufrufmöglichkeiten. Neue Backendnamen
nur einführen, wenn sie eine klar unterschiedliche Auswahl darstellen.
Nutze eine gemeinsame Fähigkeitserkennung für API, GUI und Benchmarks, damit
angebotene Optionen und tatsächlicher Dispatch nicht auseinanderlaufen.

Zeige angeforderten und verwendeten Pfad, effektive Threads, numerischen Vertrag
und Fallbackgrund verständlich an. Kennzeichne Experimente. Ein gespeicherter
GPU-Wunsch muss auf einem Rechner ohne GPU kontrolliert auf einen verfügbaren
korrekten Pfad wechseln können. Erhalte die Ergebnisstruktur, Poolnamen,
ADC-Labels, gespeicherte Sequenzinformationen und Export-/Notebook-Kompatibilität.

Ändere den Standard erst nach breiter Genauigkeits- und Leistungsmessung.
Verhindere insbesondere, dass ein Backend in der GUI aktiviert erscheint,
während der Lauf vollständig über den bisherigen NumPy-Fallback ausgeführt wird.
Dokumentiere den tatsächlichen Anteil gemischter Ausführung, soweit relevant.

**Gemeinsame Test- und Benchmarkmatrix**

| Dimension | Mindestens abzudeckende Fälle |
| --- | --- |
| Physik | Keine Konversion, heterogene Konversion, nahezu gleiche Zerfallsraten, Relaxation, analytisch prüfbarer Zufluss, Konzentration/Polarisation, Vorlauf, verzögerter Beginn, zusätzliche Pools |
| Felder | Tx eins, konstantes komplexes Tx, Nullstellen, räumliche Amplitude/Phase, statisches und dynamisches B0, Gradienten, heterogene Relaxationsparameter |
| Beobachtung | ADC am Anfang/Ende, gemeinsame Ereignisgrenzen, mehrere Spulen, Poolausgabe, Checkpoints, Demodulation, idealer Crusher und physikalisches Spoiling |
| Größe | Wenige Spins, etwa 4.096, 32.768 und mehr als 131.072 Gesamtspins; ein Spin und mehrere Subspins pro Elternvoxel, insbesondere $3\times3\times3$ |
| Sequenz | Kleiner analytischer Fall, FID/FLASH, EPI/ADC-lastiger Readout, CSI, kurzer und langer spektralselektiver bSSFP-Zug sowie vorhandene Spektral-Raum-Fälle |
| Ausführung | Seriell, mehrere Threadzahlen, wechselnde Chunk-Größen, knappe Speicherbudgets, Wiederholung, Abbruch, unsupported backend, Fallback und gespeicherte Einstellungen |

Verwende gezielte Kombinationen statt eines unnötig vollständigen kartesischen
Produkts. Ergänze jedoch mehrere anspruchsvolle Kombinationsfälle, etwa räumliches
B1 plus Zufluss plus dynamisches B0 plus Subvoxel-Spoiling. Referenzgleichheit
allein reicht nicht: teste analytische Lösungen, Massenerhaltung ohne Relaxation,
Rotationsnormen, erwartete Dephasierung und Empfangsgewichtung.

Konvergenz muss separat geprüft werden. Der aktuelle Compiler verwendet für RF
mindestens dessen natives Raster; ein kleinerer `simulation_timestep_s` erzeugt
unter Umständen exakt denselben Plan. Prüfe deshalb die tatsächlich erzeugten
Intervalle. Nutze für eine feinere unabhängige Referenz einen gezielt verfeinerten
Testplan oder Referenzintegrator, ohne die Produktionsdiskretisierung unbemerkt
zu verändern. Prüfe außerdem räumliche Subvoxelkonvergenz unabhängig vom Backend.

Erweitere vorrangig die bestehenden Tests in
[test_dynamic_phantom.py](../tests/test_dynamic_phantom.py),
[test_sequence_simulation.py](../tests/test_sequence_simulation.py),
[test_b1_fields.py](../tests/test_b1_fields.py),
[test_memory_limits.py](../tests/test_memory_limits.py),
[test_timestep_convergence.py](../tests/test_timestep_convergence.py),
[test_sequence_convergence_suite.py](../tests/test_sequence_convergence_suite.py)
und [test_benchmarks.py](../tests/test_benchmarks.py). Lege neue Dateien nur für
klar abgrenzbare Plan-, Reduktions- oder Backendverträge an. Mocktests dürfen
Fallbacklogik testen, aber keine Hardwaregenauigkeit oder Leistung behaupten.

Mit dem zuvor geprüften Interpreter sind beispielsweise folgende Prüfungen möglich:

```bash
python -m benchmarks.benchmark_acceleration_review --output /tmp/bloch_baseline.json
python -m pytest tests/test_dynamic_phantom.py tests/test_sequence_simulation.py tests/test_memory_limits.py
python -m pytest tests/test_b1_fields.py tests/test_timestep_convergence.py tests/test_sequence_convergence_suite.py tests/test_benchmarks.py
python -m benchmarks.benchmark_kernels --help
python scripts/validate_dynamic_precision.py --help
python scripts/run_test_suite.py
```

Verwende für den Gesamtlauf den vorhandenen Qt-sicheren Batchrunner und dessen
aktuelle Optionen. Führe lange Konvergenz- und Hardwareläufe gesondert aus und
protokolliere übersprungene Fälle mit Grund. Harte Zeitlimits gehören nicht in
gewöhnliche Unit-Tests; reproduzierbare Laufzeitvergleiche gehören in Benchmarks.

**Reihenfolge, Zwischenstände und Definition von fertig**

Bearbeite Phase 0 und 1 zuerst, danach B1. Entwickle Plan, nativen Treiber,
Chunking und feste Reduktion zusammenhängend in kleinen prüfbaren Schritten.
Optimiere erst danach verbleibende CPU-Kosten. Affine Propagatoren und GPU-Pfade
bauen auf diesen Verträgen auf. API-/GUI-Integration kann begleitend erfolgen;
eine Standardfreigabe folgt erst am Ende.

Führe im Repository eine kompakte Fortschrittsdatei, die pro Phase Implementierung,
relevante Dateien, bestandene Prüfungen, Messartefakte und offene Einschränkungen
festhält. Das erleichtert die Fortsetzung über mehrere Sessions. Kennzeichne
Phasen als implementiert und geprüft, experimentell, begründet verworfen oder
durch konkret benannte externe Voraussetzungen blockiert. Erfinde keine Ergebnisse,
wenn eine reale Referenzdatei oder GPU nicht vorhanden ist.

Liefere abschließend:

- Änderungen am Solver und aussagekräftige Regressionstests; vorhandene
  Referenzpfade und bisherige lokale Arbeit bleiben erhalten.
- Reproduzierbare Vorher-/Nachher-Messungen als JSON/CSV mit Hardware,
  Build-/Versionsangaben, Physik, Diskretisierung, tatsächlichem Backend,
  Laufzeitstreuung, Speicherverbrauch und vollständigen Fehlermaßen.
- Aktualisierte Anwender- und Entwicklerdokumentation zu Backendwahl,
  Genauigkeitsverträgen, Speicherplanung, Fallbacks und Einschränkungen.
- Eine klare Bewertung jeder Phase: welcher Engpass wurde entfernt, welcher
  Zeitgewinn ist gemessen, welche Genauigkeit ist belegt, was bleibt offen.

Die CPU-Umsetzung ist erfolgreich, wenn räumliches B1 und große Subvoxelobjekte
den schnellen Pfad nicht mehr aus den bisherigen vermeidbaren Gründen verlieren,
die Intervallfortschreibung samt ADC weitgehend nativ läuft, der Speicherbedarf
begrenzt ist und alle numerischen Verträge eingehalten werden. GPU- und
Propagatorverfahren gelten nur bei nachgewiesenem zusätzlichen End-to-End-Nutzen
als Produktionsbeschleunigung. Präzision hat bei jeder dieser Entscheidungen Vorrang.
