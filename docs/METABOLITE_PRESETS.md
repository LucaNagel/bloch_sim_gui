# Metabolite peak presets

The Phantom Designer's **Common metabolite** menu adds an editable peak for
the currently selected nucleus. The menu follows the explicit nucleus setting;
with **Auto**, it shows $^1$H presets for a static phantom and $^{13}$C presets
when the hyperpolarized pyruvate/lactate model is enabled. Presets are currently
curated for $^1$H, $^{13}$C, and $^{31}$P.

These are starting values, not universal tissue constants. Relaxation depends
on field strength, tissue, temperature, molecular environment, isotope
labelling, and acquisition. The simulator uses apparent $T_2^*$ to create a
Lorentzian linewidth, while many publications report intrinsic $T_2$. Directly
measured or linewidth-derived $T_2^*$ values are used where available; other
rows are explicitly marked as estimates in the menu. Every preset adds one
Lorentzian component, so coupled multiplets require additional manually edited
peaks.

## Proton presets

The $^1$H list covers common brain-MRS signals. The main resonances are based on
published human-brain assignments, and NAA, creatine, and choline use 3 T
relaxation/linewidth measurements. The broader metabolite selection follows a
published in-vivo basis set. See the
[3 T and 7 T relaxation study](https://doi.org/10.4172/2155-9937.S1-002),
[3 T brain-metabolite study](https://pubmed.ncbi.nlm.nih.gov/11477653/), and
[high-field metabolite basis set](https://pmc.ncbi.nlm.nih.gov/articles/PMC2843548/).
Additional shifts come from the
[35-metabolite spectral reference](https://pubmed.ncbi.nlm.nih.gov/10861994/).

| Preset | Shift (ppm) | $T_1$ (s) | $T_2^*$ (ms) | Basis |
|---|---:|---:|---:|---|
| NAA | 2.01 | 1.38 | 60.3 | Human brain, 3 T; $T_2^*$ from linewidth |
| Creatine | 3.03 | 1.38 | 78.4 | Human brain, 3 T; $T_2^*$ from linewidth |
| Choline | 3.22 | 1.06 | 74.0 | Human brain, 3 T; $T_2^*$ from linewidth |
| Glutamate | 2.35 | 1.41 | 52 | Representative 3 T $T_1$; estimated $T_2^*$ |
| Glutamine | 2.45 | 1.04 | 52 | Representative 3 T $T_1$; estimated $T_2^*$ |
| myo-Inositol | 3.56 | 1.20 | 52 | Editable 3 T starting estimate |
| GABA | 3.01 | 1.30 | 52 | Editable 3 T starting estimate |
| Lactate | 1.33 | 1.50 | 52 | Editable 3 T starting estimate |
| NAAG | 2.04 | 1.38 | 52 | Editable 3 T starting estimate |
| Alanine | 1.48 | 1.30 | 52 | Editable 3 T starting estimate |
| Aspartate | 2.82 | 1.30 | 52 | Editable 3 T starting estimate |
| Ascorbate | 3.73 | 1.30 | 52 | Editable 3 T starting estimate |
| Glucose | 3.43 | 1.20 | 52 | Editable 3 T starting estimate |
| Glutathione | 2.95 | 1.30 | 52 | Edited-MRS target; editable estimate |
| Glycine | 3.55 | 1.30 | 52 | Overlaps myo-inositol; editable estimate |
| Glycerophosphocholine | 3.23 | 1.06 | 52 | Total-choline component; editable estimate |
| Phosphocholine | 3.22 | 1.06 | 52 | Total-choline component; editable estimate |
| Phosphoethanolamine | 3.98 | 1.30 | 52 | Editable 3 T starting estimate |
| scyllo-Inositol | 3.35 | 1.20 | 52 | Editable 3 T starting estimate |
| Taurine | 3.42 | 1.30 | 52 | Editable 3 T starting estimate |
| 2-Hydroxyglutarate | 2.25 | 1.30 | 52 | IDH-mutant glioma marker; editable estimate |

## Carbon-13 presets

The chemical shifts in this table are the project-supplied values and are kept
exactly in `METABOLITE_CS_PPM`. They cover the commonly imaged hyperpolarized
[1-$^{13}$C]pyruvate products, fumarate-to-malate and aspartate experiments,
and urea perfusion imaging. The relaxation defaults draw on human 3 T
pyruvate/lactate/bicarbonate measurements, a fumarate study, and a 3 T urea
study. See the [human 3 T dynamic $T_2^*$ study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10872504/),
[human brain dynamic MRS study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8776582/),
[multi-compound relaxation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2885774/),
[fumarate-to-malate study](https://doi.org/10.1021/jacs.9b10094),
[aspartate study](https://doi.org/10.3389/fphys.2021.792769), and the
[$^{13}$C urea study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4011557/).
The expanded probe list also uses primary studies of
[$^{13}$C alpha-ketoglutarate and 2-hydroxyglutarate](https://pmc.ncbi.nlm.nih.gov/articles/PMC3908661/)
and [$^{13}$C dehydroascorbate/vitamin C](https://pmc.ncbi.nlm.nih.gov/articles/PMC3219134/).

| Preset | Shift (ppm) | $T_1$ (s) | $T_2^*$ (ms) | Basis |
|---|---:|---:|---:|---|
| Pyruvate | 171.076 | 30 | 75.56 | Human 3 T model $T_1$; early human-brain $T_2^*$ |
| Lactate | 183.35 | 24 | 30 | Human in-vivo 3 T $T_1$ and human-brain $T_2^*$ |
| Alanine | 176.5 | 25 | 30 | Editable estimate |
| Pyruvate hydrate | 179.5 | 30 | 30 | Editable estimate |
| Fumarate | 175.4 | 28 | 50 | Reported $T_1$; estimated $T_2^*$ |
| Malate C1 | 181.7 | 28 | 50 | Editable fumarate-experiment estimate |
| Malate C4 | 180.5 | 28 | 50 | Editable fumarate-experiment estimate |
| Bicarbonate | 161.0 | 25.5 | 108.17 | Human in-vivo 3 T |
| Urea | 163.5 | 46 | 90 | 3 T aqueous [$^{13}$C]urea; measured $T_2$ used as an upper-bound $T_2^*$ start |
| Carbon dioxide | 124.5 | 44.7 | 50 | Solution $T_1$ at 11.7 T; estimated $T_2^*$ |
| Aspartate C1 | 176.92 | 25 | 30 | Editable estimate |
| Aspartate C4 | 180.20 | 25 | 30 | Editable estimate |
| alpha-Ketoglutarate C1 | 172.6 | 52 | 50 | Solution $T_1$ at 3 T; estimated $T_2^*$ |
| alpha-Ketoglutarate hydrate | 180.9 | 54 | 50 | Solution $T_1$ at 3 T; estimated $T_2^*$ |
| 2-Hydroxyglutarate C1 | 183.9 | 26 | 50 | Single-sample solution $T_1$ at 3 T; estimated $T_2^*$ |
| Dehydroascorbate C1 | 174.0 | 56.5 | 50 | Solution $T_1$ at 3 T; estimated $T_2^*$ |
| Vitamin C C1 | 177.8 | 29.2 | 50 | Buffered-solution $T_1$ at 3 T; estimated $T_2^*$ |

The pyruvate $T_2^*$ value is deliberately an early-time value. Human dynamic
measurements found that it fell from about 75.6 ms to 22.2 ms during the brain
acquisition, which is a useful reminder to tune the preset to the experiment.

## Phosphorus-31 presets

$^{31}$P MRS commonly resolves energy and phospholipid-metabolism peaks. Shifts
are referenced to phosphocreatine at 0 ppm. The assignments come from a human
brain $^{31}$P basis set, and the relaxation values use human-brain 3 T
measurements with $T_2^*$ calculated as $1/(\pi\,\mathrm{FWHM})$. See
the [human-brain basis-set study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4438275/)
and [3 T relaxation study](https://doi.org/10.1002/nbm.4169). The NAD and
UDP-glucose additions use a
[human-brain 7 T study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4772768/).

| Preset | Shift (ppm) | $T_1$ (s) | $T_2^*$ (ms) |
|---|---:|---:|---:|
| Phosphocreatine | 0.00 | 2.66 | 43.6 |
| Inorganic phosphate | 4.84 | 1.84 | 20.5 |
| Phosphoethanolamine | 6.77 | 3.42 | 14.0 |
| Phosphocholine | 6.23 | 3.42 | 14.0 |
| Glycerophosphoethanolamine | 3.49 | 3.44 | 14.1 |
| Glycerophosphocholine | 2.94 | 2.72 | 14.1 |
| gamma-ATP | -2.53 | 0.80 | 16.5 |
| alpha-ATP | -7.56 | 0.88 | 18.5 |
| beta-ATP | -16.18 | 0.89 | 10.6 |
| Extracellular inorganic phosphate | 5.24 | 3.82 | 20.5 |
| NAD+ | -8.31 | 2.07 | 2.5 |
| NADH | -8.13 | 2.07 | 2.5 |
| UDP-glucose | -9.72 | 2.95 | 3.1 |
| Membrane phospholipids | 2.30 | 2.50 | 14.0 |

$^{19}$F and $^{23}$Na remain manual: $^{19}$F MRS normally targets a selected
exogenous compound, while the $^{23}$Na resonance is not a metabolite-specific
menu in the same sense. This avoids attaching arbitrary biological names and
relaxation values to those nuclei.
