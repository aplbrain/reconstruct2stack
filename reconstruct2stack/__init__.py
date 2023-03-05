from functools import cached_property
from typing import Optional, Union
from typing import Tuple, Union, List, Dict, Any
import json
import pathlib
import numpy as np
import cv2


class Contour:
    """

    ['color', 'closed', 'negative', 'x', 'y', 'hidden', 'mode', 'tags', 'history', 'name']
    """

    def __init__(
        self,
        points: Optional[Union[np.ndarray, List[Tuple[float, float]]]] = None,
        x: Optional[Union[np.ndarray, List[float]]] = None,
        y: Optional[Union[np.ndarray, List[float]]] = None,
        color: Tuple[float, float, float] = None,
        closed: bool = None,
        negative: bool = None,
        hidden: bool = None,
        name: str = None,
        mode: int = None,
        tags: List[str] = None,
        history: List[str] = None,
        offset_xy: Union[Tuple[float, float], None] = None,
    ):
        """
        Create a new contour.

        """
        if points is not None:
            self.points = np.array(points)
        elif x is not None and y is not None:
            self.points = np.array(list(zip(x, y)))
        else:
            raise ValueError("Must provide either points or x and y.")
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]
        self.color = color
        self.name = name
        self.hidden = hidden
        self.closed = closed
        self.negative = negative
        self.tags = tags
        self.history = history
        self.mode = mode
        self.offset_xy = offset_xy

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """
        Create a new contour from a dictionary.

        """
        return cls(
            x=d["x"],
            y=d["y"],
            color=d["color"],
            closed=d["closed"],
            negative=d["negative"],
            hidden=d["hidden"],
            name=d["name"],
            mode=d["mode"],
            tags=d["tags"],
            history=d["history"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the contour.

        """
        return {
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "closed": self.closed,
            "negative": self.negative,
            "hidden": self.hidden,
            "name": self.name,
            "mode": self.mode,
            "tags": self.tags,
            "history": self.history,
        }

    def __len__(self) -> int:
        """
        Return the number of points in the contour.

        """
        return len(self.points)

    def __repr__(self) -> str:
        """
        Return a string representation of the contour.

        """
        return f"<Contour '{self.name}' (len={len(self)})>"

    def copy(self) -> "Contour":
        """
        Return a copy of the contour.

        """
        return Contour.from_dict(self.to_dict())

    def with_updated(self, **kwargs) -> "Contour":
        """
        Return a new contour with the given attributes updated.

        """
        return Contour.from_dict({**self.to_dict(), **kwargs})

    def with_mag(self, mag: float) -> "Contour":
        """
        Return a new contour with the points scaled by a given magnitude.

        """
        return self.copy().with_updated(x=self.x * mag, y=self.y * mag)

    def with_tforms(
        self, tforms: Tuple[float, float, float, float, float, float]
    ) -> "Contour":
        """
        Return a new contour with the points transformed.

        tforms is a 6-tuple of floats. This can be reshaped into the affine
        transformation matrix and applied to the points.

        """
        # return Contour(points=np.matmul(self.points, np.array(tforms).reshape(2,3).T))
        return self.copy().with_updated(
            points=np.matmul(self.points, np.array(tforms).reshape(3, 2).T)
        )


class JSERIngester:
    def __init__(self, jser: Union[str, pathlib.Path, Dict[str, Any]]):
        """
        Create a new JSER file ingest.

        Can read from a file, a path, or a dictionary.

        """
        if isinstance(jser, (str, pathlib.Path)):
            jser = json.loads(pathlib.Path(jser).read_text())
        self.jser = jser

    @cached_property
    def _prefix(self) -> str:
        """
        Return the prefix of the JSER file.

        """
        return ".".join(self.raw.keys().__iter__().__next__().split(".")[:-1])

    @cached_property
    def raw(self) -> Dict[str, Any]:
        """
        Return the raw JSER dictionary.

        """
        return self.jser

    def __len__(self) -> int:
        """
        Return the number of segments.

        """
        return len(self.raw.keys())

    def __getitem__(self, key: Union[int, str]) -> Dict[str, Any]:
        """
        Return the segment with the given key.

        """
        if isinstance(key, int):
            key = f"{self._prefix}.{key}"
        return self.raw[key]

    def _normalize_key(self, key: Union[int, str]) -> int:
        """
        Return the key as an integer.

        """
        if isinstance(key, str):
            key = int(key.split(".")[-1])
        return key

    def keys(self) -> List[int]:
        """
        Return the keys of the raw JSER dictionary.

        """
        return list(
            [int(k.split(".")[-1]) for k in self.raw.keys() if not k.endswith(".ser")]
        )

    def get_unique_colors_for_slice(self, slice_num: int) -> List[str]:
        """
        Return the colors for the given slice.

        """
        colors = []
        for c in self.get_raw_contours_for_slice(slice_num):
            if c.color not in colors:
                colors.append(c.color)
        return colors

    def get_all_unique_colors(self) -> List[str]:
        """
        Return the colors for the given slice.

        """
        colors = []
        for z in self.keys():
            for c in self.get_raw_contours_for_slice(z):
                if c.color not in colors:
                    colors.append(c.color)
        return colors

    def get_raw_contours_for_slice(self, slice_num: int) -> List[Contour]:
        """
        Return the contours for the given slice without performing affines.

        """
        slice_contours = self[slice_num]["contours"]
        flattened_contours = []
        for name, contours in slice_contours.items():
            for contour in contours:
                contour["name"] = name
                flattened_contours.append(contour)
        return [Contour.from_dict(d) for d in flattened_contours]

    def contours(
        self,
        key: Union[int, str],
        filter_by_colors: Optional[List[List[float]]] = None,
    ) -> List[Contour]:
        """
        Return the contours for the given segment.

        """
        key = self._normalize_key(key)
        filter_by_colors = (
            [list(int(f) for f in c) for c in filter_by_colors]
            if filter_by_colors is not None
            else None
        )
        return [
            d.with_mag(1.0 / self[key]["mag"]).with_tforms(
                self[key]["tforms"]["default"]
            )
            for d in self.get_raw_contours_for_slice(key)
            if filter_by_colors is None or d.color in filter_by_colors
        ]
