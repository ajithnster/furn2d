"""Shared settings for the showroom image pipeline."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader: KEY=value lines; real environment variables win."""
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_dotenv()
STORE = "https://veronahomefurniture.com"
USER_AGENT = "Mozilla/5.0 (showroom-pipeline)"


@dataclass(frozen=True)
class Spot:
    """Where a product category sits in the room, in room pixels.

    `roi` bounds the area the extractor searches for the new object (x0, y0, x1, y1).
    `hint` is the placement instruction given to the image model.
    """
    roi: tuple[int, int, int, int]
    hint: str


@dataclass(frozen=True)
class Room:
    id: str
    image: Path
    size: tuple[int, int]
    spots: dict[str, Spot]


LIVING_ROOM = Room(
    id="living-room",
    image=ROOT / "rooms/living-room/empty-room.webp",
    size=(1535, 1024),
    spots={
        # Matches the live showroom's sofa layers (x≈500-1500, y≈450-930).
        "sofa": Spot(
            roi=(380, 380, 1535, 1000),
            hint=(
                "standing on the wooden floor in the middle-right of the room, in front of "
                "the plain white wall and the right-hand window, roughly centred horizontally "
                "between x=55% and x=60% of the image width, its seat front facing the camera "
                "at the same three-quarter angle as the product photo"
            ),
        ),
    },
)

OUT = ROOT / "out"
CACHE = ROOT / "cache"

# Rooms created by `pipeline room --room <id> --create` (empty photo generated in the living
# room's style; spots are unused by the asset/demo flow, which positions pieces in the browser).
BEDROOM = Room(id="bedroom", image=ROOT / "rooms/bedroom/empty-room.webp", size=(1536, 1024), spots={})
DINING_ROOM = Room(id="dining", image=ROOT / "rooms/dining/empty-room.webp", size=(1536, 1024), spots={})
ROOMS = {r.id: r for r in (LIVING_ROOM, BEDROOM, DINING_ROOM)}
