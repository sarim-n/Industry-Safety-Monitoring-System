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

    # Face-only / Head-only person detection aspect ratio threshold
    # Face crops typically have aspect ratio (height / width) < 1.0 (width >= height)
    max_face_only_aspect_ratio: float = 1.00


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


def box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def is_helmet_region_observable(
    p_box: BBox,
    img_width: float,
    img_height: float,
    head_roi: BBox,
    config: PPEAssociationConfig
) -> bool:
    """
    Determines whether the head/top-of-person region required to observe a helmet
    is sufficiently visible within the image frame.
    
    Observability criteria for Helmet:
    1. Top frame boundary truncation check: p_box.y1 must not touch or extend past top edge.
    2. Face-only / head-only crop check: person box aspect ratio (height / width) must
       indicate body/head context rather than a face-only crop (aspect_ratio >= max_face_only_aspect_ratio).
    3. Person width must be >= min_person_width_px.
    4. Head ROI visible height inside frame must be >= min_head_height_px.
    """
    # 1. Top boundary truncation check
    if p_box.y1 <= config.boundary_margin_px:
        return False

    # 2. Face-only / head-only crop check via aspect ratio
    aspect_ratio = p_box.height / max(1.0, p_box.width)
    if aspect_ratio < config.max_face_only_aspect_ratio:
        return False

    # 3. Minimum person width check
    if p_box.width < config.min_person_width_px:
        return False

    # 4. Visible Head ROI height inside canvas
    vis_head_y1 = max(0.0, head_roi.y1)
    vis_head_y2 = min(float(img_height), head_roi.y2)
    vis_head_height = max(0.0, vis_head_y2 - vis_head_y1)

    if vis_head_height < config.min_head_height_px:
        return False

    return True


def is_mask_region_observable(
    p_box: BBox,
    img_width: float,
    img_height: float,
    face_roi: BBox,
    config: PPEAssociationConfig
) -> bool:
    """
    Determines whether the face region required to observe a mask is sufficiently
    visible within the image frame.
    
    Observability criteria for Mask:
    1. Face ROI vertical extent must be inside frame boundaries.
    2. Person width must be >= min_person_width_px.
    3. Visible Face ROI height inside frame must be >= min_face_height_px.
    """
    # 1. Vertical boundary truncation of face ROI
    if face_roi.y1 < config.boundary_margin_px or face_roi.y2 > (float(img_height) - config.boundary_margin_px):
        return False

    # 2. Minimum person width check
    if p_box.width < config.min_person_width_px:
        return False

    # 3. Visible Face ROI height inside canvas
    vis_face_y1 = max(0.0, face_roi.y1)
    vis_face_y2 = min(float(img_height), face_roi.y2)
    vis_face_height = max(0.0, vis_face_y2 - vis_face_y1)

    if vis_face_height < config.min_face_height_px:
        return False

    return True


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

    # Phase 7.6 Diagnostics
    head_roi_bbox: Optional[Tuple[float, float, float, float]] = None
    face_roi_bbox: Optional[Tuple[float, float, float, float]] = None
    head_observable: bool = True
    face_observable: bool = True
    helmet_candidate_count: int = 0
    mask_candidate_count: int = 0
    uncertain_reason: Optional[str] = None

    def get_debug_summary(self) -> str:
        w = max(0.0, self.person_bbox[2] - self.person_bbox[0])
        h = max(0.0, self.person_bbox[3] - self.person_bbox[1])
        h_cand = self.helmet_candidate_count
        m_cand = self.mask_candidate_count
        h_score = f"{self.helmet_association_score:.2f}" if self.helmet_association_score is not None else "N/A"
        m_score = f"{self.mask_association_score:.2f}" if self.mask_association_score is not None else "N/A"
        return (
            f"Worker #{self.person_index} | bbox=({self.person_bbox[0]:.1f},{self.person_bbox[1]:.1f},{self.person_bbox[2]:.1f},{self.person_bbox[3]:.1f}) "
            f"w={w:.1f} h={h:.1f} | Head Obs={self.head_observable} Face Obs={self.face_observable} | "
            f"Helmet Cand={h_cand} Score={h_score} State={self.helmet_detected} | "
            f"Mask Cand={m_cand} Score={m_score} State={self.mask_detected} | "
            f"Status={self.safety_status} (Reason: {self.uncertain_reason})"
        )



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

            # 4. Assess Head / Face Visibility (PPE Observability)
            head_obs = is_helmet_region_observable(p_box, img_width, img_height, head_roi, self.config)
            face_obs = is_mask_region_observable(p_box, img_width, img_height, face_roi, self.config)

            head_vis = "VISIBLE" if head_obs else "CROPPED"
            face_vis = "VISIBLE" if face_obs else "CROPPED"

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
                safety_status="UNCERTAIN",
                head_roi_bbox=(head_roi.x1, head_roi.y1, head_roi.x2, head_roi.y2),
                face_roi_bbox=(face_roi.x1, face_roi.y1, face_roi.x2, face_roi.y2),
                head_observable=head_obs,
                face_observable=face_obs
            )
            person_states.append(state)
            person_head_rois.append(head_roi)
            person_face_rois.append(face_roi)

        # 5. Calculate Helmet Association Scores & Match Deterministically
        helmet_candidates = []  # (score, person_idx, helmet_idx)
        per_person_helmet_cand_counts = [0] * len(persons)

        for p_idx, p in enumerate(persons):
            p_box = p['box']
            head_roi = person_head_rois[p_idx]

            for h_idx, h in enumerate(helmets):
                h_box = h['box']

                inter = compute_intersection(h_box, head_roi)
                ioh = inter / h_box.area if h_box.area > 0 else 0.0

                horiz_dist_norm = abs(h_box.center_x - p_box.center_x) / max(1.0, p_box.width)
                horiz_score = max(0.0, 1.0 - horiz_dist_norm)

                vert_dist_norm = (h_box.center_y - p_box.y1) / max(1.0, p_box.height)
                if -0.25 <= vert_dist_norm <= 0.40:
                    vert_score = 1.0 - abs(vert_dist_norm - 0.05) / 0.35
                else:
                    vert_score = 0.0

                total_score = 0.50 * ioh + 0.30 * horiz_score + 0.20 * vert_score

                if total_score >= self.config.min_helmet_association_score and ioh > 0.10:
                    helmet_candidates.append((total_score, p_idx, h_idx))
                    per_person_helmet_cand_counts[p_idx] += 1

        for idx, count in enumerate(per_person_helmet_cand_counts):
            person_states[idx].helmet_candidate_count = count

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
        per_person_mask_cand_counts = [0] * len(persons)

        for p_idx, p in enumerate(persons):
            p_box = p['box']
            face_roi = person_face_rois[p_idx]

            for m_idx, m in enumerate(masks):
                m_box = m['box']

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
                    per_person_mask_cand_counts[p_idx] += 1

        for idx, count in enumerate(per_person_mask_cand_counts):
            person_states[idx].mask_candidate_count = count

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
            # Handle Helmet status if not YES (Associated PPE takes priority!)
            if state.helmet_detected != "YES":
                if state.head_visibility == "CROPPED":
                    state.helmet_detected = "UNKNOWN"
                else:
                    state.helmet_detected = "NO"

            # Handle Mask status if not YES (Associated PPE takes priority!)
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
                state.uncertain_reason = None
            elif h_det == "NO" and m_det == "YES":
                state.safety_status = "NO_HELMET"
                state.uncertain_reason = None
            elif h_det == "YES" and m_det == "NO":
                state.safety_status = "NO_MASK"
                state.uncertain_reason = None
            elif h_det == "NO" and m_det == "NO":
                state.safety_status = "NO_HELMET_AND_MASK"
                state.uncertain_reason = None
            else:
                state.safety_status = "UNCERTAIN"
                if h_det == "UNKNOWN" and m_det == "UNKNOWN":
                    state.uncertain_reason = "BOTH_REGIONS_UNOBSERVABLE"
                elif h_det == "UNKNOWN":
                    state.uncertain_reason = "HELMET_REGION_UNOBSERVABLE"
                else:
                    state.uncertain_reason = "MASK_REGION_UNOBSERVABLE"

        return person_states

