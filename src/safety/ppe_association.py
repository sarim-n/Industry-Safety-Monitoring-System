"""
PPE Association Module for Industrial Safety Monitoring System
================================================================
Associates detected helmets and masks with detected persons using spatial geometry,
performs image-boundary and visibility assessment, and derives per-person safety status.

Authoritative classes:
  0 = helmet
  1 = mask
  2 = person
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class PPEAssociationConfig:
    """Configurable parameters for PPE spatial association and visibility evaluation."""
    # Detection confidence thresholds
    person_conf: float = 0.50
    helmet_conf: float = 0.25
    mask_conf: float = 0.20

    # Head and Face ROI ratios relative to person bbox [x1, y1, x2, y2]
    head_region_top_offset_ratio: float = -0.15  # Allow helmet slightly above person y1
    head_region_height_ratio: float = 0.35       # Top 35% of person bbox height
    
    face_region_top_ratio: float = 0.08          # Starts near y1 + 0.08 * height
    face_region_height_ratio: float = 0.32       # Height is 0.32 * height
    face_region_width_ratio: float = 0.70        # Central 70% width of person bbox

    # Spatial Association minimum score thresholds
    min_helmet_association_score: float = 0.20
    min_mask_association_score: float = 0.20

    # Boundary margin in pixels to detect image edge contact
    boundary_margin_px: float = 3.0

    # Minimum pixel size for head/face region visibility
    min_head_height_px: float = 10.0
    min_face_height_px: float = 8.0
    min_person_width_px: float = 12.0


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


def compute_intersection(boxA: BBox, boxB: BBox) -> float:
    x1 = max(boxA.x1, boxB.x1)
    y1 = max(boxA.y1, boxB.y1)
    x2 = min(boxA.x2, boxB.x2)
    y2 = min(boxA.y2, boxB.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    return inter_w * inter_h


@dataclass
class PersonPPEState:
    person_index: int
    person_bbox: Tuple[float, float, float, float]
    person_confidence: float

    # Boundary flags
    touches_left_boundary: bool
    touches_right_boundary: bool
    touches_top_boundary: bool
    touches_bottom_boundary: bool

    # Visibility states ("VISIBLE", "CROPPED", "UNKNOWN")
    head_visibility: str
    face_visibility: str

    # Helmet details
    helmet_detected: str  # "YES", "NO", "UNKNOWN"

    # Mask details
    mask_detected: str    # "YES", "NO", "UNKNOWN"

    # Optional Helmet details
    helmet_bbox: Optional[Tuple[float, float, float, float]] = None
    helmet_confidence: Optional[float] = None
    helmet_association_score: Optional[float] = None

    # Optional Mask details
    mask_bbox: Optional[Tuple[float, float, float, float]] = None
    mask_confidence: Optional[float] = None
    mask_association_score: Optional[float] = None

    # Final Derived Safety Status
    safety_status: str = "UNCERTAIN"



class PPEAssociator:
    """Engine for performing deterministic spatial PPE association and safety evaluation."""

    def __init__(self, config: Optional[PPEAssociationConfig] = None):
        self.config = config or PPEAssociationConfig()

    def process_detections(
        self,
        detections: List[Dict[str, Any]],
        img_width: float,
        img_height: float
    ) -> List[PersonPPEState]:
        """
        Process raw YOLO detections for a single image.
        
        Args:
            detections: List of dicts with keys: 'cls' (int), 'conf' (float), 'box' ([x1, y1, x2, y2])
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of PersonPPEState for all detected persons.
        """
        # Filter detections by class-specific confidence thresholds
        persons = []
        helmets = []
        masks = []

        for d in detections:
            c = d['cls']
            conf = d['conf']
            box = BBox(*d['box'])

            if c == 2 and conf >= self.config.person_conf:
                persons.append({'box': box, 'conf': conf, 'raw': d})
            elif c == 0 and conf >= self.config.helmet_conf:
                helmets.append({'box': box, 'conf': conf, 'raw': d, 'assigned': False})
            elif c == 1 and conf >= self.config.mask_conf:
                masks.append({'box': box, 'conf': conf, 'raw': d, 'assigned': False})

        # Sort persons deterministically by x1 coordinate then confidence descending
        persons.sort(key=lambda p: (p['box'].x1, -p['conf']))

        # Prepare person states and ROIs
        person_states = []
        person_head_rois = []
        person_face_rois = []

        for idx, p in enumerate(persons):
            p_box = p['box']
            p_conf = p['conf']

            # 1. Boundary awareness check
            touches_left = p_box.x1 <= self.config.boundary_margin_px
            touches_right = p_box.x2 >= (img_width - self.config.boundary_margin_px)
            touches_top = p_box.y1 <= self.config.boundary_margin_px
            touches_bottom = p_box.y2 >= (img_height - self.config.boundary_margin_px)

            # 2. Define Head ROI
            head_y1 = max(0.0, p_box.y1 + self.config.head_region_top_offset_ratio * p_box.height)
            head_y2 = p_box.y1 + self.config.head_region_height_ratio * p_box.height
            head_roi = BBox(p_box.x1, head_y1, p_box.x2, head_y2)

            # 3. Define Face ROI
            face_w_margin = (1.0 - self.config.face_region_width_ratio) / 2.0
            face_x1 = p_box.x1 + face_w_margin * p_box.width
            face_x2 = p_box.x2 - face_w_margin * p_box.width
            face_y1 = p_box.y1 + self.config.face_region_top_ratio * p_box.height
            face_y2 = p_box.y1 + (self.config.face_region_top_ratio + self.config.face_region_height_ratio) * p_box.height
            face_roi = BBox(face_x1, face_y1, face_x2, face_y2)

            # 4. Assess Head / Face Visibility
            # Top boundary touching can indicate head truncation
            if touches_top and (p_box.y1 <= 2.0 or head_roi.height < self.config.min_head_height_px):
                head_vis = "CROPPED"
            elif p_box.width < self.config.min_person_width_px:
                head_vis = "CROPPED"
            else:
                head_vis = "VISIBLE"

            if touches_top and (face_y1 <= 2.0 or face_roi.height < self.config.min_face_height_px):
                face_vis = "CROPPED"
            elif p_box.width < self.config.min_person_width_px:
                face_vis = "CROPPED"
            else:
                face_vis = "VISIBLE"

            state = PersonPPEState(
                person_index=idx + 1,
                person_bbox=(p_box.x1, p_box.y1, p_box.x2, p_box.y2),
                person_confidence=p_conf,
                touches_left_boundary=touches_left,
                touches_right_boundary=touches_right,
                touches_top_boundary=touches_top,
                touches_bottom_boundary=touches_bottom,
                head_visibility=head_vis,
                face_visibility=face_vis,
                helmet_detected="UNKNOWN",
                mask_detected="UNKNOWN",
                safety_status="UNCERTAIN"
            )
            person_states.append(state)
            person_head_rois.append(head_roi)
            person_face_rois.append(face_roi)

        # 5. Calculate Helmet Association Scores & Match Deterministically
        helmet_candidates = []  # (score, person_idx, helmet_idx)
        for p_idx, p in enumerate(persons):
            p_box = p['box']
            head_roi = person_head_rois[p_idx]

            for h_idx, h in enumerate(helmets):
                h_box = h['box']

                # Compute Score Components
                # (a) Overlap of helmet with Head ROI (Intersection over Helmet Area)
                inter = compute_intersection(h_box, head_roi)
                ioh = inter / h_box.area if h_box.area > 0 else 0.0

                # (b) Horizontal center alignment
                horiz_dist_norm = abs(h_box.center_x - p_box.center_x) / max(1.0, p_box.width)
                horiz_score = max(0.0, 1.0 - horiz_dist_norm)

                # (c) Vertical position check (helmet center should be near top of person box)
                vert_dist_norm = (h_box.center_y - p_box.y1) / max(1.0, p_box.height)
                if -0.25 <= vert_dist_norm <= 0.40:
                    vert_score = 1.0 - abs(vert_dist_norm - 0.05) / 0.35
                else:
                    vert_score = 0.0

                total_score = 0.50 * ioh + 0.30 * horiz_score + 0.20 * vert_score

                if total_score >= self.config.min_helmet_association_score and ioh > 0.10:
                    helmet_candidates.append((total_score, p_idx, h_idx))

        # Sort candidate matches by score descending for greedy deterministic 1-to-1 matching
        helmet_candidates.sort(key=lambda x: x[0], reverse=True)

        assigned_persons_helmet = set()
        assigned_helmets = set()

        for score, p_idx, h_idx in helmet_candidates:
            if p_idx not in assigned_persons_helmet and h_idx not in assigned_helmets:
                assigned_persons_helmet.add(p_idx)
                assigned_helmets.add(h_idx)
                h_info = helmets[h_idx]
                h_box = h_info['box']
                state = person_states[p_idx]
                state.helmet_detected = "YES"
                state.helmet_bbox = (h_box.x1, h_box.y1, h_box.x2, h_box.y2)
                state.helmet_confidence = h_info['conf']
                state.helmet_association_score = score

        # 6. Calculate Mask Association Scores & Match Deterministically
        mask_candidates = []  # (score, person_idx, mask_idx)
        for p_idx, p in enumerate(persons):
            p_box = p['box']
            face_roi = person_face_rois[p_idx]

            for m_idx, m in enumerate(masks):
                m_box = m['box']

                # Compute Score Components
                inter = compute_intersection(m_box, face_roi)
                iom = inter / m_box.area if m_box.area > 0 else 0.0

                horiz_dist_norm = abs(m_box.center_x - p_box.center_x) / max(1.0, p_box.width)
                horiz_score = max(0.0, 1.0 - horiz_dist_norm)

                vert_dist_norm = (m_box.center_y - p_box.y1) / max(1.0, p_box.height)
                if 0.05 <= vert_dist_norm <= 0.45:
                    vert_score = 1.0 - abs(vert_dist_norm - 0.22) / 0.25
                else:
                    vert_score = 0.0

                total_score = 0.50 * iom + 0.30 * horiz_score + 0.20 * vert_score

                if total_score >= self.config.min_mask_association_score and iom > 0.10:
                    mask_candidates.append((total_score, p_idx, m_idx))

        mask_candidates.sort(key=lambda x: x[0], reverse=True)

        assigned_persons_mask = set()
        assigned_masks = set()

        for score, p_idx, m_idx in mask_candidates:
            if p_idx not in assigned_persons_mask and m_idx not in assigned_masks:
                assigned_persons_mask.add(p_idx)
                assigned_masks.add(m_idx)
                m_info = masks[m_idx]
                m_box = m_info['box']
                state = person_states[p_idx]
                state.mask_detected = "YES"
                state.mask_bbox = (m_box.x1, m_box.y1, m_box.x2, m_box.y2)
                state.mask_confidence = m_info['conf']
                state.mask_association_score = score

        # 7. Resolve Final PPE States & Safety Decision Logic
        for state in person_states:
            # Handle Helmet status if not YES
            if state.helmet_detected != "YES":
                if state.head_visibility == "CROPPED":
                    state.helmet_detected = "UNKNOWN"
                else:
                    state.helmet_detected = "NO"

            # Handle Mask status if not YES
            if state.mask_detected != "YES":
                if state.face_visibility == "CROPPED":
                    state.mask_detected = "UNKNOWN"
                else:
                    state.mask_detected = "NO"

            # Apply Strict Safety Matrix
            h_det = state.helmet_detected
            m_det = state.mask_detected

            if h_det == "YES" and m_det == "YES":
                state.safety_status = "SAFE"
            elif h_det == "NO" and m_det == "YES":
                state.safety_status = "NO_HELMET"
            elif h_det == "YES" and m_det == "NO":
                state.safety_status = "NO_MASK"
            elif h_det == "NO" and m_det == "NO":
                state.safety_status = "NO_HELMET_AND_MASK"
            else:
                # Any UNKNOWN state results in UNCERTAIN (never false safety violation)
                state.safety_status = "UNCERTAIN"

        return person_states
