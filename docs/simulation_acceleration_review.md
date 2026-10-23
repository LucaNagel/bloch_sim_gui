# Beschleunigung der Bloch-Simulation

Stand: 27.09.2026. Grundlage sind der untersuchte Arbeitsstand des Quellcodes,
vorhandene Validierungsberichte und ein eigener kleiner CPU-Vergleich vom 26.09.2026.

**Die größten kurzfristigen Chancen liegen bei B1-Unterstützung in den nativen
Kerneln und bei einer durchgehend kompilierten dynamischen Simulation.** Eine GPU
passt grundsätzlich gut zur unabhängigen Berechnung vieler Spins. Der Exchange
erfordert hier keinen Datenaustausch zwischen Rechenkernen: Pyruvat und Laktat
werden gemeinsam innerhalb desselben Voxels fortgeschrieben. Für Apple Metal ist
gegenwärtig vor allem die numerische Genauigkeit die Hürde.

**Wie die Magnetisierung berechnet wird**

Die ereignisbasierte Simulation übersetzt RF, Gradienten, ADC und angeforderte
Zwischenzustände zunächst in Zeitintervalle. Dynamische Kurven liefern zusätzliche
Grenzen. Während RF wird die Wellenform gerastert; RF-freie Abschnitte behalten die
erforderlichen Ereignis- und Messgrenzen. RF und Gradienten werden über das jeweilige
Intervall gemittelt. Der konfigurierte RF-Zeitschritt kann das native RF-Raster
vergröbern; ein kleinerer Wert verfeinert es derzeit nicht unter dessen Rasterweite.
Die Magnetisierung wird anschließend rekursiv von Intervall zu Intervall berechnet.
Sie wird bei der Sequenzsimulation nur an ADC-Zeitpunkten zum Signal summiert und
an ausgewählten Checkpoints vollständig gespeichert.
Quelle: [Sequenzcompiler](../src/blochsimulator/sequence/compiler.py).

Beim **gewöhnlichen Bloch-Solver** gilt für die normierte Magnetisierung schematisch

$$
\mathbf M_{n+1}=D(\Delta t)R_n\mathbf M_n+\mathbf b(\Delta t),
\qquad
D=\operatorname{diag}(E_2,E_2,E_1),\quad
E_i=e^{-\Delta t/T_i},\quad
\mathbf b=(0,0,1-E_1)^{\mathsf T}.
$$

Die Rotation berücksichtigt RF einschließlich komplexer Tx-Sensitivität, B0,
chemische Verschiebung und Gradienten. Rotation und Relaxation erfolgen nacheinander.
Der optimierte Kernel vermeidet die explizite Rotationsmatrix und nutzt besondere
RF-freie Schritte. Statische spektrale Komponenten werden unabhängig simuliert und
ihre Signale anschließend addiert.

Der **dynamische Solver** hält pro Spin für jeden Pool drei
Magnetisierungskomponenten sowie gegebenenfalls Konzentrationen. Ein Intervall
besteht aus freier Entwicklung über eine halbe Zeitspanne, RF-Rotation über den
ganzen Schritt und einer zweiten freien Halbzeitspanne. Das ist ein symmetrisches
Operator-Splitting. Transversal wird ein vorbereiteter komplexer Faktor angewendet:

$$
M_{+,p}'=M_{+,p}\exp\!\left(-h/T_{2,p}-2\pi i\int_t^{t+h}f_p(\tau)\,d\tau\right),
\qquad M_{+,p}=M_{x,p}+iM_{y,p}.
$$

Longitudinal wird die irreversible Umwandlung analytisch mit Exponential- und
Quelltermkoeffizienten integriert. Mit $R_1=1/T_1$, ohne Zufluss und bei Relaxation
gegen null gilt

$$
\frac{d}{dt}
\begin{pmatrix}M_{z,P}\\M_{z,L}\end{pmatrix}
= \begin{pmatrix}-(R_{1,P}+k_{PL})&0\\k_{PL}&-R_{1,L}\end{pmatrix}
\begin{pmatrix}M_{z,P}\\M_{z,L}\end{pmatrix}.
$$

Zufluss wird abschnittsweise über seine Randwerte berücksichtigt; bei
Konzentrationsverfolgung wird zusätzlich die Gleichgewichtspolarisation einbezogen.
**Der aktuelle Exchange wirkt nur longitudinal; transversaler kohärenter Austausch
und Laktat-Rückkonversion sind darin nicht enthalten.** Das muss bei einer Portierung
erhalten bleiben. Analytische Teilschritte machen die gesamte RF-Simulation wegen
Wellenformdiskretisierung und Splitting noch nicht exakt.
Quelle: [dynamischer Solver](../src/blochsimulator/dynamic_phantom.py), insbesondere
`_free_step`, `_longitudinal_step` und `simulate_dynamic_sequence`.

**Vorhandene Kernel**

| Pfad | Implementierung und Parallelisierung |
| --- | --- |
| Ältere Bloch-/Phantom-API | C über Cython; seriell oder OpenMP über Positionen/Frequenzen bzw. Voxel. Gruppierte Gewebe teilen Relaxationsfaktoren. Der Verlaufsmodus speichert alle Zeitpunkte und kann viel Speicher benötigen. |
| Sequenz-`reference` / `optimized` | Beide in C; ein Worker verfolgt einen Spin durch die komplette Sequenz. Voxelweise Speicherblöcke, threadlokale ADC-Summen; optimiert mit Quaternion-Rotation, RF-freiem Sonderfall und bis zu 64 exakten Relaxationsgruppen. |
| Dynamisch `reference` | Python-Zeitschleife, NumPy-Operationen und viele temporäre Arrays; Vergleichsreferenz. |
| Dynamisch `optimized` | Weiterhin Python/NumPy, aber wiederverwendete Zustände, Arbeitsarrays und begrenzte Koeffizientencaches. Standard in API und GUI; die CPU-Thread-Einstellung parallelisiert diese Zeitschleife nicht automatisch. |
| Dynamisch `native_serial` / `native_parallel` | Strikte Float64-Cython-Kernel für Teiloperationen. OpenMP über Spins ab 1.024 simulierten Spins. Im unterstützten Modell laufen bis zu 512 RF-Intervalle gemeinsam in einem Block; beide Pools bleiben dabei beim selben Worker. |
| Dynamisch `metal_hybrid` | Experimentelle Apple-GPU-Berechnung mit Float64-CPU-Stichproben und unabhängiger Validierung. Bei Nichtunterstützung oder fehlgeschlagener Prüfung wird vollständig auf CPU gerechnet. |

Quellen: [C-Kernel](../src/blochsimulator/bloch_core_modified.c),
[Cython-Kernel](../src/blochsimulator/dynamic_bloch_wrapper.pyx),
[Simulator und Auswahl](../src/blochsimulator/simulator.py),
[Metal-Anbindung](../src/blochsimulator/dynamic_metal_backend.py).

**Konkrete Engpässe**

- **B1 schaltet den schnellen dynamischen RF-Pfad ab.** Die Prüfung lautet effektiv
  „alle Tx-Werte exakt eins“. Selbst eine konstante Karte mit Faktor 0,9 fällt in
  `_rf_rotate_spatial`: pro RF-Intervall ortsabhängige Sinus-/Kosinusberechnung,
  temporäre Arrays und Synchronisierung der beiden Zustandsdarstellungen.
  Native longitudinale Schritte können weiterlaufen; die RF-Blockfusion entfällt.
- **Die dynamische Zeitschleife ist nur teilweise kompiliert.** Außerhalb der
  fusionierten RF-Blöcke bleiben viele Array-Durchläufe, Python-Aufrufe,
  Zuflussauswertungen und gegebenenfalls OpenMP-Synchronisationen pro Intervall.
  ADC-Auswertung erfolgt weiterhin über NumPy-Summen. Viele ADC-Punkte und
  Empfangsspulen erhöhen diesen Aufwand.
- **Große Spin-Zahlen verursachen zusätzlich einen Pfadwechsel.** Der dynamische
  Solver expandiert alle Subvoxel gleichzeitig. Ein $3\times3\times3$-Raster
  multipliziert Arbeit und Zustände mit 27. Oberhalb von 131.072 Spins entfällt
  derzeit die RF-Blockfusion. Weitere Ausschlüsse sind dynamisches B0,
  ortsabhängige Zuflussverzögerungen und nicht unterstützte Zuflussmodelle.
  Mehr als zwei Pools führen vollständig zum NumPy-Pfad; zusätzliche
  Produktkonversionen berechnen manche Koeffizienten erneut.
- **Auch kleine Rechenschritte haben Kosten jenseits des Matrixprodukts.**
  Koeffizientenberechnung bei Cache-Misses, Speicherverkehr und Signalreduktion
  können dominieren. Mehr Kerne helfen dann nur begrenzt. Maßgeblich sind aktive
  Spins, Zeitintervalle und ADC-Zahl; die Rekonstruktionsmatrix allein beschreibt
  die Rechenlast nicht.

**Eigene Kontrollmessung**

Auf diesem Apple-Silicon-Rechner: 4.096 aktive Spins, zwei Pools mit Umwandlung,
Zufluss und Konzentrationsverfolgung, acht RF-Pulse mit insgesamt 512 RF-Intervallen,
256 ADC-Werte und insgesamt 776 Intervalle, Float64. Median aus drei Läufen nach
Aufwärmen, einschließlich Sequenzkompilierung; BLAS-Threadlimit auf eins gesetzt.

| Kernel | Tx überall 1 | Tx überall 0,9 | Räumliche Tx-Karte |
| --- | ---: | ---: | ---: |
| `optimized` | 266 ms | 258 ms | 250 ms |
| `native_serial` | 104 ms | 201 ms | 185 ms |
| `native_parallel`, 4 Threads | 95 ms | 238 ms | 222 ms |
| `native_parallel`, 8 Threads | 109 ms | 289 ms | 286 ms |

Mit Tx eins wurden alle 512 RF-Intervalle in acht Blöcke fusioniert, sonst keines.
Im Profil der räumlichen Karte entfielen rund 57 von 258 ms auf die 512 Aufrufe
von `_rf_rotate_spatial`. Die maximal gemessene relative L2-Abweichung von Signal,
Poolsignal und Endzustand gegenüber der jeweiligen Referenz lag bei
$3{,}2\times10^{-17}$. Die B1-Fälle haben unterschiedliche Physik; die Tabelle zeigt
den Pfadwechsel, keinen isolierten kausalen B1-Zeitaufschlag. Der kurze Versuch
belegt keine Beschleunigungsrate für große Produktionssequenzen.
[Messdaten und Profile](simulation_acceleration_benchmark.json),
[Reproduktionsskript](../benchmarks/benchmark_acceleration_review.py).

**GPU-Potenzial und Präzision**

Eine geeignete GPU-Aufteilung wäre ein Thread pro Spin mit beiden Pools und mehreren
aufeinanderfolgenden Zeitschritten. Zustände bleiben auf der GPU; ADC-Beiträge werden
blockweise mit festgelegter Summationsreihenfolge reduziert. Ein GPU-Aufruf und
CPU-Rücktransfer nach jedem kleinen Zeitschritt würde unnötigen Aufwand erzeugen.
Ein bloßer Austausch von NumPy durch eine GPU-Arraybibliothek schöpft dieses Potenzial
daher voraussichtlich nicht aus.

Metal bietet keinen `double`-Datentyp für diese Shader.
[Apple-Spezifikation, Abschnitt 2.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
Die im Repository dokumentierten M3-Experimente scheiterten bereits an der
provisorischen Signal-NRMSE-Grenze $10^{-3}$: nach fünf Volumina etwa
$7{,}0\times10^{-3}$ insgesamt und $4{,}8\times10^{-2}$ für Laktat. Die Grenze
$10^{-3}$ ist experimentell und keine bestätigte Toleranz für deine Anwendung. Ein Zustand aus
zwei Float32-Zahlen verbesserte das kaum. Das spricht dafür, dass auch die
Koeffizienten-/Phasenberechnung genauer werden muss. Diese historischen Messungen
wurden hier nicht erneut ausgeführt.
Quelle: [Metal-Machbarkeitsbericht](dynamic_metal_feasibility.md).

Eine GPU mit nativer Float64-Unterstützung, beispielsweise über CUDA, eröffnet einen
anderen Weg. Die konkrete Float64-Rechenleistung muss zum Zielgerät passen;
gleicher Datentyp allein garantiert zudem keine bitidentischen Ergebnisse bei
anderer Operations- oder Reduktionsreihenfolge.
[NVIDIA zur Gleitkommaarithmetik](https://docs.nvidia.com/cuda/floating-point/index.html).
Der vorhandene Hybrid ist keine allgemeine Genauigkeitsgarantie: Sein Gate prüft
geschätzte Signalfehler, während Fehler des Endzustands zwar ausgewiesen werden,
aber derzeit nicht in die Freigabe eingehen. Ein CPU-Fallback kann insgesamt sogar
mehr Zeit kosten als eine direkte CPU-Simulation.

**Empfohlene Reihenfolge**

1. **B1 in Float64 beschleunigen:** konstante komplexe Tx-Faktoren erkennen und
   in die RF-Wellenform einrechnen; danach einen nativen Kernel für räumliche
   Tx-Karten ergänzen. Wiederkehrende RF-/B1-Koeffizienten begrenzt cachen.
   Keine näherungsweise B1-Gruppierung ohne gesonderte Validierung.
2. **Dynamische Simulation über ganze Segmente kompilieren:** RF, freie
   Entwicklung, Kinetik und ADC integrieren; Voxelblöcke über die vollständige
   Sequenz führen. Das reduziert Speicherverkehr und Synchronisation und hebt
   die heutige Gesamtspin-Grenze der RF-Fusion auf. Checkpoints, Abbruchfähigkeit
   und deterministische Signalreduktion müssen erhalten bleiben.
3. **Danach einen Float64-GPU-Prototyp am verfügbaren Zielgerät messen.** Den
   bisherigen Metal-Pfad nur nach weiterer Genauigkeitsarbeit ausbauen. Als
   weitergehende CPU-/GPU-Option lassen sich wiederholte RF-Blöcke zu affinen
   Propagatoren zusammenfassen, sofern ihre Parameter und Beobachtungsgrenzen
   passen; deren Vorbereitung, Speicherbedarf und Rundungsfehler separat prüfen.
4. **Jeden Schritt gegen Präzision absichern:** vollständige lange Sequenzen,
   getrennte Pyruvat-/Laktatsignale, Phasen, schwache Signale, Endzustände und
   Rekonstruktionen vergleichen; zusätzlich Zeit- und Subvoxelkonvergenz sowie
   Grenzfälle mit verschwindender Konversion und nahezu gleichen Zerfallsraten testen.
   Eine Referenzübereinstimmung bei gleicher Diskretisierung ersetzt diese
   Konvergenzprüfung nicht. Der dynamische Cython-Code wird bereits ohne Fast Math
   und ohne FMA-Kontraktion gebaut; die gewöhnliche C-Erweiterung verwendet laut
   [Build-Konfiguration](../setup.py) auf macOS/Linux weiterhin `-ffast-math`.
   Für eine belastbare Referenz dieses Pfads sollte ein strikter Build hinzukommen.
