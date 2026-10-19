"""Read ParaVision JCAMP-DX parameter files and scanner template metadata.

ParaVision stores every parameter file (``subject``, ``acqp``, ``method``,
``visu_pars`` ...) as JCAMP-DX 4.24 text.  This module parses those files and
extracts the identity metadata of a real scanner dataset (owner, operator,
institution, subject and study UIDs, reference frequencies).  The Bruker
exporter copies that metadata into simulated datasets so ParaVision and the
usual analysis tools accept them next to measured data.

Only the parameter files are read.  Raw data, images and adjustment results of
the template are never copied.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np


_HEADER_KEYS = ("TITLE", "JCAMPDX", "DATATYPE", "ORIGIN", "OWNER")
_DIMENSION_RE = re.compile(r"^\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)")
_REPEAT_RE = re.compile(r"^@(\d+)\*\((.*)\)$", re.S)
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

# Subject parameters whose names changed in ParaVision 360 3.x.  The first
# name of every tuple is the ParaVision 5-7 name.
SUBJECT_PARAMETER_ALIASES = {
    "date": ("SUBJECT_date", "SUBJECT_study_date"),
    "abs_date": ("SUBJECT_abs_date", "SUBJECT_study_abs_date"),
    "referral": ("SUBJECT_referral", "SUBJECT_study_operator"),
    "comment": ("SUBJECT_comment", "SUBJECT_study_comment"),
    "sex": ("SUBJECT_sex", "SUBJECT_gender"),
    "weight": ("SUBJECT_weight", "SUBJECT_study_weight"),
    "position": ("SUBJECT_position", "SUBJECT_study_instrument_position"),
    "entry": ("SUBJECT_entry",),
}

# Subject parameters describing the RF coil, gradient and shim hardware.  They
# are copied verbatim when the template coil configuration is kept.
SUBJECT_HARDWARE_PARAMETERS = (
    "SUBJECT_study_adj_config",
    "SUBJECT_study_gradient_system",
    "SUBJECT_study_shim_system",
    "SUBJECT_study_coil_configuration",
    "SUBJECT_study_coil_table",
    "SUBJECT_study_coils",
)


@dataclass(frozen=True)
class JcampFile:
    """Parsed JCAMP-DX parameter file.

    ``parameters`` maps every ``##$NAME`` to its raw value text.  Use
    :meth:`get` for a parsed Python value and :meth:`text` for strings and
    enum values.
    """

    header: dict[str, str]
    parameters: dict[str, str]
    comments: tuple[str, ...] = ()
    path: Optional[Path] = None

    def __contains__(self, name: str) -> bool:
        return name in self.parameters

    def raw(self, name: str) -> Optional[str]:
        return self.parameters.get(name)

    def get(self, name: str, default=None):
        raw = self.parameters.get(name)
        if raw is None:
            return default
        return parse_jcamp_value(raw)

    def text(self, name: str, default: str = "") -> str:
        value = self.get(name)
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                value = value[0]
            else:
                return " ".join(str(item) for item in value)
        if isinstance(value, np.ndarray):
            return " ".join(str(item) for item in value.reshape(-1).tolist())
        return str(value)

    def number(self, name: str) -> Optional[float]:
        value = self.get(name)
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0] if value.size else None
        elif isinstance(value, (list, tuple)):
            value = value[0] if value else None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @property
    def paravision_version(self) -> Optional[str]:
        return detect_paravision_version(self)

    def comment_path(self) -> Optional[str]:
        """Return the scanner path recorded in the ``$$ /...`` header line."""
        for comment in self.comments:
            if comment.startswith("/"):
                return comment.strip()
        return None

    def comment_user_host(self) -> Optional[str]:
        """Return ``user@host`` from the ``$$ <timestamp>  user@host`` line."""
        for comment in self.comments:
            match = re.match(
                r"^\d{4}-\d{2}-\d{2} [^ ]+ [+-]\d{4}\s+(\S+@\S+)$", comment
            )
            if match:
                return match.group(1)
        return None

    def comment_process(self) -> Optional[str]:
        for comment in self.comments:
            if comment.startswith("process "):
                return comment[len("process ") :].strip()
        return None


def read_jcamp(path) -> JcampFile:
    path = Path(path)
    text = path.read_bytes().decode("latin-1")
    parsed = parse_jcamp(text)
    return replace(parsed, path=path)


def parse_jcamp(text: str) -> JcampFile:
    header: dict[str, str] = {}
    parameters: dict[str, str] = {}
    comments: list[str] = []
    current_name: Optional[str] = None
    current_lines: list[str] = []
    is_parameter = False

    def flush():
        if current_name is None:
            return
        value = "\n".join(current_lines).strip()
        if is_parameter:
            parameters[current_name] = value
        else:
            header[current_name] = value

    for line in text.splitlines():
        if line.startswith("$$"):
            comment = line[2:].strip()
            if not comment.startswith("@vis="):
                comments.append(comment)
            continue
        if line.startswith("##"):
            flush()
            body = line[2:]
            is_parameter = body.startswith("$")
            if is_parameter:
                body = body[1:]
            name, separator, value = body.partition("=")
            if not separator:
                current_name = None
                current_lines = []
                continue
            current_name = name.strip()
            current_lines = [value]
            if current_name == "END":
                current_name = None
            continue
        if current_name is not None:
            current_lines.append(line)
    flush()
    return JcampFile(
        header={key: value for key, value in header.items()},
        parameters=parameters,
        comments=tuple(comments),
    )


def parse_jcamp_value(raw: str):
    """Convert a raw JCAMP value into ``int``, ``float``, ``str``, list or array.

    ``( n )`` dimension prefixes followed by data become numeric arrays with
    that shape, or a string for character arrays.  Bare parenthesised
    structures become tuples.
    """

    text = str(raw).strip()
    match = _DIMENSION_RE.match(text)
    if match is not None and text[match.end() :].strip():
        shape = tuple(int(value) for value in match.group(1).split(","))
        values = _tokenize(text[match.end() :])
        expanded = _expand_values(values)
        if expanded and all(isinstance(value, str) for value in expanded):
            if len(expanded) == 1:
                return expanded[0]
            return list(expanded)
        if expanded and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in expanded
        ):
            array = np.asarray(expanded)
            if int(np.prod(shape)) == array.size:
                return array.reshape(shape)
            return array
        return list(expanded)
    tokens = _expand_values(_tokenize(text))
    if len(tokens) == 1:
        return tokens[0]
    return list(tokens)


def _tokenize(text: str) -> list:
    tokens: list = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == "<":
            end = index + 1
            chars = []
            while end < length and text[end] != ">":
                if text[end] != "\n" and text[end] != "\r":
                    chars.append(text[end])
                end += 1
            tokens.append(_StringToken("".join(chars)))
            index = end + 1
            continue
        if char == "(":
            depth = 0
            end = index
            in_string = False
            while end < length:
                current = text[end]
                if current == "<":
                    in_string = True
                elif current == ">":
                    in_string = False
                elif not in_string and current == "(":
                    depth += 1
                elif not in_string and current == ")":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            inner = text[index + 1 : end]
            tokens.append(tuple(_parse_struct(inner)))
            index = end + 1
            continue
        end = index
        while end < length and not text[end].isspace():
            if text[end] == "(" and text[index] == "@":
                depth = 0
                while end < length:
                    if text[end] == "(":
                        depth += 1
                    elif text[end] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    end += 1
            end += 1
        tokens.append(text[index:end])
        index = end
    return tokens


def _parse_struct(inner: str) -> list:
    parts: list[str] = []
    depth = 0
    in_string = False
    start = 0
    for position, char in enumerate(inner):
        if char == "<":
            in_string = True
        elif char == ">":
            in_string = False
        elif not in_string and char == "(":
            depth += 1
        elif not in_string and char == ")":
            depth -= 1
        elif not in_string and depth == 0 and char == ",":
            parts.append(inner[start:position])
            start = position + 1
    parts.append(inner[start:])
    values = []
    for part in parts:
        tokens = _expand_values(_tokenize(part))
        if len(tokens) == 1:
            values.append(tokens[0])
        else:
            values.append(tuple(tokens))
    return values


class _StringToken(str):
    pass


def _expand_values(tokens: Iterable) -> list:
    values: list = []
    for token in tokens:
        if isinstance(token, _StringToken):
            values.append(str(token))
            continue
        if isinstance(token, tuple):
            values.append(token)
            continue
        match = _REPEAT_RE.match(token)
        if match is not None:
            repeated = _expand_values(_tokenize(match.group(2)))
            values.extend(repeated * int(match.group(1)))
            continue
        values.append(_scalar(token))
    return values


def _scalar(token: str):
    if _NUMBER_RE.match(token):
        if re.match(r"^[+-]?\d+$", token):
            return int(token)
        return float(token)
    return token


def detect_paravision_version(jcamp: JcampFile) -> Optional[str]:
    """Return a normalized ParaVision version such as ``7.0.0`` or ``360.3.6``."""

    title = jcamp.header.get("TITLE", "")
    match = re.search(r"ParaVision\s+360\s+V?(\d+(?:\.\d+)*)", title)
    if match:
        return f"360.{match.group(1)}"
    match = re.search(r"ParaVision\s+(\d+(?:\.\d+)*)", title)
    if match:
        return match.group(1)
    software = jcamp.text("ACQ_sw_version")
    match = re.search(r"PV[- ]?360[.-]?(\d+(?:\.\d+)*)", software)
    if match:
        return f"360.{match.group(1)}"
    match = re.search(r"PV[- ]?(\d+(?:\.\d+)*)", software)
    if match:
        return match.group(1)
    creator = jcamp.text("VisuCreatorVersion")
    if re.match(r"^\d+(\.\d+)*$", creator):
        return creator
    return None


@dataclass(frozen=True)
class BrukerTemplateMetadata:
    """Scanner identity metadata copied from a measured ParaVision dataset.

    Every field is editable in the GUI before export.  Empty strings mean
    "use the exporter default".  Acquisition parameters (TR, TE, matrix,
    bandwidth ...) are never taken from the template: they always describe
    the simulated sequence.
    """

    source_path: str = ""
    paravision_version: str = ""
    owner: str = ""
    user_host: str = ""
    data_root: str = ""
    process_root: str = ""
    operator: str = ""
    institution: str = ""
    station: str = ""
    system_order_number: str = ""
    manufacturer: str = ""
    subject_id: str = ""
    subject_name: str = ""
    subject_uid: str = ""
    subject_type: str = ""
    subject_sex: str = ""
    subject_birth_date: str = ""
    subject_weight_kg: Optional[float] = None
    study_name: str = ""
    study_uid: str = ""
    study_number: int = 1
    referring_physician: str = ""
    subject_entry: str = ""
    subject_position: str = ""
    study_date: str = ""
    coil_receive_name: str = ""
    coil_transmit_name: str = ""
    reference_frequencies_mhz: tuple[tuple[str, float], ...] = ()
    hardware_parameters: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def reference_frequency_mhz(self, nucleus: str) -> Optional[float]:
        wanted = normalize_nucleus(nucleus)
        for name, frequency in self.reference_frequencies_mhz:
            if normalize_nucleus(name) == wanted:
                return float(frequency)
        return None

    def with_updates(self, **values) -> "BrukerTemplateMetadata":
        return replace(self, **values)


def normalize_nucleus(value: str) -> str:
    text = str(value or "").strip().upper().replace("-", "")
    aliases = {"H1": "1H", "1H": "1H", "C13": "13C", "13C": "13C"}
    if text in aliases:
        return aliases[text]
    match = re.match(r"^([A-Z][A-Z]?)(\d+)$", text)
    if match:
        return f"{match.group(2)}{match.group(1).capitalize()}"
    match = re.match(r"^(\d+)([A-Z][A-Z]?)$", text)
    if match:
        return f"{match.group(1)}{match.group(2).capitalize()}"
    return str(value or "").strip()


def find_bruker_dataset_files(path) -> dict[str, Path]:
    """Locate subject, acqp, method and visu_pars files around ``path``.

    ``path`` may be a study folder (contains ``subject``), a scan folder
    (contains ``acqp``), a ``pdata/<n>`` folder or any of the parameter files.
    Finder duplicates such as ``acqp 2`` are ignored.
    """

    target = Path(path).expanduser()
    if target.is_file():
        target = target.parent
    if not target.is_dir():
        raise FileNotFoundError(f"Bruker template folder not found: {target}")

    files: dict[str, Path] = {}
    scan_dir: Optional[Path] = None
    study_dir: Optional[Path] = None
    if (target / "visu_pars").is_file() and (target.parent.name == "pdata"):
        files["pdata_visu_pars"] = target / "visu_pars"
        scan_dir = target.parent.parent
    elif (target / "acqp").is_file() or (target / "method").is_file():
        scan_dir = target
    elif (target / "subject").is_file():
        study_dir = target
    else:
        scans = _scan_directories(target)
        if not scans:
            raise FileNotFoundError(
                f"No ParaVision subject or acqp file found in {target}"
            )
        study_dir = target
    if scan_dir is not None and study_dir is None:
        if (scan_dir.parent / "subject").is_file():
            study_dir = scan_dir.parent
    if scan_dir is None and study_dir is not None:
        scans = _scan_directories(study_dir)
        scan_dir = scans[0] if scans else None

    if study_dir is not None and (study_dir / "subject").is_file():
        files["subject"] = study_dir / "subject"
    if scan_dir is not None:
        for name in ("acqp", "method", "visu_pars"):
            candidate = scan_dir / name
            if candidate.is_file():
                files[name] = candidate
        if "pdata_visu_pars" not in files:
            pdata = sorted(
                (
                    item
                    for item in (scan_dir / "pdata").glob("*")
                    if item.is_dir() and item.name.isdigit()
                ),
                key=lambda item: int(item.name),
            )
            for proc in pdata:
                if (proc / "visu_pars").is_file():
                    files["pdata_visu_pars"] = proc / "visu_pars"
                    break
    if study_dir is not None:
        files["_study_dir"] = study_dir
    if scan_dir is not None:
        files["_scan_dir"] = scan_dir
    if not any(not key.startswith("_") for key in files):
        raise FileNotFoundError(f"No ParaVision parameter files found in {target}")
    return files


def _scan_directories(study_dir: Path) -> list[Path]:
    scans = [
        item
        for item in study_dir.iterdir()
        if item.is_dir() and item.name.isdigit() and (item / "acqp").is_file()
    ]
    return sorted(scans, key=lambda item: int(item.name))


def load_bruker_template(path) -> BrukerTemplateMetadata:
    """Read scanner identity metadata from a measured ParaVision dataset."""

    files = find_bruker_dataset_files(path)
    subject = read_jcamp(files["subject"]) if "subject" in files else None
    acqp = read_jcamp(files["acqp"]) if "acqp" in files else None
    visu = None
    for key in ("pdata_visu_pars", "visu_pars"):
        if key in files:
            visu = read_jcamp(files[key])
            break
    sources = [item for item in (acqp, subject, visu) if item is not None]
    if not sources:
        raise FileNotFoundError(f"No ParaVision parameter files found in {path}")

    version = ""
    for item in sources:
        detected = item.paravision_version
        if detected:
            version = detected
            break
    owner = next((item.header.get("OWNER", "") for item in sources), "")
    user_host = next(
        (item.comment_user_host() for item in sources if item.comment_user_host()),
        "",
    )
    process = next(
        (item.comment_process() for item in sources if item.comment_process()), ""
    )
    process_root = ""
    if process:
        marker = process.find("/prog/")
        process_root = process[:marker] if marker > 0 else ""

    data_root = ""
    if subject is not None and subject.comment_path():
        data_root = str(Path(subject.comment_path()).parent.parent)
    elif acqp is not None and acqp.comment_path():
        data_root = str(Path(acqp.comment_path()).parent.parent.parent)

    def subject_text(key: str) -> str:
        if subject is None:
            return ""
        for name in SUBJECT_PARAMETER_ALIASES[key]:
            if name in subject:
                return subject.text(name)
        return ""

    def first_text(*pairs) -> str:
        for source, name in pairs:
            if source is not None and name in source:
                value = source.text(name)
                if value:
                    return value
        return ""

    weight = None
    if subject is not None:
        for name in SUBJECT_PARAMETER_ALIASES["weight"]:
            if name in subject:
                weight = subject.number(name)
                break
    if weight is None and visu is not None:
        weight = visu.number("VisuSubjectWeight")

    study_number = 1
    if subject is not None and subject.number("SUBJECT_study_nr") is not None:
        study_number = int(subject.number("SUBJECT_study_nr"))
    elif visu is not None and visu.number("VisuStudyNumber") is not None:
        study_number = int(visu.number("VisuStudyNumber"))

    hardware = ()
    if subject is not None:
        hardware = tuple(
            (name, subject.raw(name))
            for name in SUBJECT_HARDWARE_PARAMETERS
            if name in subject
        )

    return BrukerTemplateMetadata(
        source_path=str(files.get("_study_dir") or files.get("_scan_dir") or path),
        paravision_version=version,
        owner=owner,
        user_host=user_host,
        data_root=data_root,
        process_root=process_root,
        operator=first_text((acqp, "ACQ_operator"), (subject, "SUBJECT_referral")),
        institution=first_text((acqp, "ACQ_institution"), (visu, "VisuInstitution")),
        station=first_text((acqp, "ACQ_station"), (visu, "VisuStation")),
        system_order_number=first_text(
            (visu, "VisuSystemOrderNumber"), (acqp, "ACQ_system_order_number")
        ),
        manufacturer=first_text((visu, "VisuManufacturer"))
        or next((item.header.get("ORIGIN", "") for item in sources), ""),
        subject_id=first_text((subject, "SUBJECT_id"), (visu, "VisuSubjectId")),
        subject_name=first_text(
            (subject, "SUBJECT_name_string"), (visu, "VisuSubjectName")
        ),
        subject_uid=first_text(
            (subject, "SUBJECT_instance_uid"),
            (subject, "patient_instance_uid"),
            (visu, "VisuSubjectUid"),
        ),
        subject_type=first_text((subject, "SUBJECT_type"), (visu, "VisuSubjectType")),
        subject_sex=subject_text("sex") or first_text((visu, "VisuSubjectSex")),
        subject_birth_date=first_text(
            (subject, "SUBJECT_dbirth"), (visu, "VisuSubjectBirthDate")
        ),
        subject_weight_kg=weight,
        study_name=first_text((subject, "SUBJECT_study_name"), (visu, "VisuStudyId")),
        study_uid=first_text(
            (subject, "SUBJECT_study_instance_uid"), (visu, "VisuStudyUid")
        ),
        study_number=study_number,
        referring_physician=subject_text("referral")
        or first_text((visu, "VisuStudyReferringPhysician")),
        subject_entry=subject_text("entry"),
        subject_position=subject_text("position")
        or first_text((visu, "VisuSubjectPosition")),
        study_date=subject_text("date") or first_text((visu, "VisuStudyDate")),
        coil_receive_name=first_text((visu, "VisuCoilReceiveName")),
        coil_transmit_name=first_text((visu, "VisuCoilTransmitName")),
        reference_frequencies_mhz=_reference_frequencies(files),
        hardware_parameters=hardware,
    )


def _reference_frequencies(files: dict[str, Path]) -> tuple[tuple[str, float], ...]:
    """Collect ``NUC1``/``BF1`` pairs of every scan in the template study."""

    scan_dirs: list[Path] = []
    study_dir = files.get("_study_dir")
    if study_dir is not None:
        scan_dirs = _scan_directories(study_dir)
    elif files.get("_scan_dir") is not None:
        scan_dirs = [files["_scan_dir"]]
    frequencies: dict[str, float] = {}
    for scan_dir in scan_dirs:
        try:
            acqp = read_jcamp(scan_dir / "acqp")
        except (OSError, ValueError):
            continue
        nucleus = normalize_nucleus(acqp.text("NUC1"))
        frequency = acqp.number("BF1")
        if nucleus and frequency and nucleus not in frequencies:
            frequencies[nucleus] = frequency
    return tuple(frequencies.items())


ANONYMIZED_SUBJECT = "Anonymous"


def anonymize_template(
    template: BrukerTemplateMetadata, *, keep_operator: bool = True
) -> BrukerTemplateMetadata:
    """Remove subject-identifying fields and issue new subject/study UIDs.

    Scanner identity (owner, station, institution, system order number and
    reference frequencies) is kept because ParaVision needs it to accept the
    dataset.  The operator is kept unless ``keep_operator`` is false.
    """

    return replace(
        template,
        subject_id=ANONYMIZED_SUBJECT,
        subject_name=ANONYMIZED_SUBJECT,
        subject_uid=generate_uid(),
        subject_sex="unknown",
        subject_birth_date="",
        subject_weight_kg=None,
        study_uid=generate_uid(),
        referring_physician="" if not keep_operator else template.referring_physician,
        operator=template.operator if keep_operator else "",
        study_date="",
    )


def generate_uid() -> str:
    """Return a DICOM-conformant UUID-derived UID (``2.25.<uuid>``)."""

    return f"2.25.{uuid.uuid4().int}"


TemplateSource = Union[str, Path, BrukerTemplateMetadata, None]


def resolve_template(template: TemplateSource) -> Optional[BrukerTemplateMetadata]:
    if template is None or isinstance(template, BrukerTemplateMetadata):
        return template
    return load_bruker_template(template)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

_LINE_WIDTH = 78


class JcampEnum(str):
    """Enum value written without angle brackets (e.g. ``Head_Supine``)."""


def format_jcamp_number(value) -> str:
    """Format a number the way ParaVision writes it (``%.15g``)."""

    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if not np.isfinite(number):
        return "0"
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    text = f"{number:.15g}"
    return "0" if text in ("-0", "-0.0") else text


def _format_atom(value) -> str:
    if isinstance(value, JcampEnum):
        return str(value)
    if isinstance(value, str):
        return f"<{value}>"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (tuple,)):
        return "(" + ", ".join(_format_struct_field(item) for item in value) + ")"
    if isinstance(value, (list, np.ndarray)):
        return " ".join(
            _format_atom(item) for item in np.asarray(value, dtype=object).reshape(-1)
        )
    return format_jcamp_number(value)


def _format_struct_field(value) -> str:
    if isinstance(value, (list, np.ndarray)):
        return " ".join(
            _format_atom(item) for item in np.asarray(value, dtype=object).reshape(-1)
        )
    return _format_atom(value)


def _split_atoms(text: str) -> list[str]:
    """Split ``text`` at spaces that are not inside ``<...>`` strings."""

    atoms: list[str] = []
    current: list[str] = []
    in_string = False
    for char in text:
        if char == "<":
            in_string = True
        elif char == ">":
            in_string = False
        if char == " " and not in_string:
            if current:
                atoms.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        atoms.append("".join(current))
    return atoms


def wrap_jcamp_value(text: str, prefix: str = "") -> list[str]:
    """Wrap a value greedily at 78 characters like ParaVision's PARX writer.

    ``prefix`` (e.g. ``##$NAME=``) is joined to the first atom without a
    space.  Wrapped lines keep their trailing space; the last line has none.
    Atoms longer than the line width (long strings) stay on one line.
    """

    atoms = _split_atoms(text)
    lines: list[str] = []
    line = ""
    for index, atom in enumerate(atoms):
        if index == 0:
            line = prefix + atom
        elif len(line) + 1 + len(atom) > _LINE_WIDTH:
            lines.append(line + " ")
            line = atom
        else:
            line = f"{line} {atom}"
    lines.append(line if atoms else prefix)
    return lines


class JcampWriter:
    """Build a ParaVision JCAMP-DX parameter file.

    ``comments`` are the ``$$`` header lines written after ``##OWNER``.
    Parameters named in ``omit`` are silently skipped, so one writer routine
    can serve ParaVision versions that do not know every parameter.
    """

    def __init__(
        self,
        *,
        title: str,
        origin: str,
        owner: str,
        comments: Iterable[str] = (),
        footer: Optional[str] = None,
        omit: Iterable[str] = (),
    ):
        self._lines: list[str] = [
            f"##TITLE={title}",
            "##JCAMPDX=4.24",
            "##DATATYPE=Parameter Values",
            f"##ORIGIN={origin}",
            f"##OWNER={owner}",
        ]
        self._lines.extend(f"$$ {comment}" for comment in comments)
        self._footer = footer
        self._omit = frozenset(omit)
        self._names: list[str] = []

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._names)

    def _start(self, name: str) -> str:
        self._names.append(name)
        return f"##${name}="

    def scalar(self, name: str, value) -> "JcampWriter":
        """Write a number, bool (``Yes``/``No``) or :class:`JcampEnum`."""

        if name in self._omit:
            return self
        if isinstance(value, str) and not isinstance(value, JcampEnum):
            value = JcampEnum(value)
        self._lines.extend(wrap_jcamp_value(_format_atom(value), self._start(name)))
        return self

    def enum(self, name: str, value: str) -> "JcampWriter":
        return self.scalar(name, JcampEnum(value))

    def date(self, name: str, text: str) -> "JcampWriter":
        """Write an unsized ``<...>`` value such as an ISO date."""

        if name in self._omit:
            return self
        self._lines.append(f"{self._start(name)}<{text}>")
        return self

    def string(self, name: str, text: str, size: Optional[int] = None) -> "JcampWriter":
        """Write a char array ``( size )`` followed by ``<text>``."""

        if name in self._omit:
            return self
        text = "" if text is None else str(text)
        length = max(int(size or 0), len(text) + 1)
        self._lines.append(f"{self._start(name)}( {length} )")
        self._lines.append(f"<{text}>")
        return self

    def array(self, name: str, values, shape=None) -> "JcampWriter":
        """Write a numeric, bool or enum array with its ``( n, m )`` shape."""

        if name in self._omit:
            return self
        data = np.asarray(values, dtype=object)
        if shape is None:
            shape = data.shape if data.ndim else (1,)
        shape_text = ", ".join(str(int(item)) for item in shape)
        self._lines.append(f"{self._start(name)}( {shape_text} )")
        atoms = " ".join(_format_atom(item) for item in data.reshape(-1))
        self._lines.extend(wrap_jcamp_value(atoms))
        return self

    def enums(self, name: str, values, shape=None) -> "JcampWriter":
        return self.array(name, [JcampEnum(item) for item in values], shape)

    def strings(self, name: str, texts, size: int, shape=None) -> "JcampWriter":
        """Write a string array ``( n, size )`` followed by ``<a> <b> ...``."""

        if name in self._omit:
            return self
        texts = [str(item) for item in texts]
        width = max([int(size)] + [len(item) + 1 for item in texts])
        count = shape if shape is not None else (len(texts),)
        shape_text = ", ".join(str(int(item)) for item in tuple(count) + (width,))
        self._lines.append(f"{self._start(name)}( {shape_text} )")
        self._lines.extend(wrap_jcamp_value(" ".join(f"<{item}>" for item in texts)))
        return self

    def struct(self, name: str, fields) -> "JcampWriter":
        """Write a struct ``(a, b, ...)``.

        ``str`` fields become ``<str>``, :class:`JcampEnum` fields stay bare,
        nested tuples become nested structs and lists become space-separated
        arrays.
        """

        if name in self._omit:
            return self
        self._lines.extend(
            wrap_jcamp_value(_format_atom(tuple(fields)), self._start(name))
        )
        return self

    def structs(self, name: str, items, shape=None) -> "JcampWriter":
        """Write an array of structs ``( n )`` followed by ``(..) (..)``."""

        if name in self._omit:
            return self
        items = [tuple(item) for item in items]
        count = shape if shape is not None else (len(items),)
        shape_text = ", ".join(str(int(item)) for item in count)
        self._lines.append(f"{self._start(name)}( {shape_text} )")
        self._lines.extend(
            wrap_jcamp_value(" ".join(_format_atom(item) for item in items))
        )
        return self

    def raw(self, name: str, text: str) -> "JcampWriter":
        """Write a parameter value copied verbatim from another file."""

        if name in self._omit:
            return self
        text = str(text).rstrip("\n")
        if "\n" in text:
            first, rest = text.split("\n", 1)
            self._lines.append(f"{self._start(name)}{first}")
            self._lines.extend(rest.split("\n"))
        else:
            self._lines.append(f"{self._start(name)}{text}")
        return self

    def comment(self, text: str) -> "JcampWriter":
        self._lines.append(f"$$ {text}")
        return self

    def text(self) -> str:
        lines = list(self._lines)
        lines.append("##END=")
        if self._footer:
            lines.append(f"$$ {self._footer}")
        return "\n".join(lines) + "\n"

    def write(self, path) -> Path:
        path = Path(path)
        path.write_bytes(self.text().encode("latin-1", errors="replace"))
        return path
