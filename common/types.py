from dataclasses import dataclass


@dataclass
class Detection:
    x: int            # centre
    y: int
    r: int            # rayon du cercle englobant
    color: str        # "red" | "blue" | "yellow"
    label: str        # "limite 50", "stop", "danger", ...
    confidence: float
    shape: str = ""   # "circle" | "triangle" | "octagon" | "rect"