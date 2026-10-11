# RoomShift AI — Real Phone-Video Recording & Validation Protocol

This document establishes the standardized procedure for capturing genuine paired smartphone videos of an indoor room to validate RoomShift AI's 3D reconstruction, cross-viewpoint scene registration, and spatial change detection pipeline.

---

## 1. Objectives

1. Validate end-to-end performance on **genuine handheld smartphone video** across differing camera trajectories.
2. Evaluate cross-recording alignment under viewpoint variation and lighting stability.
3. Test change detection under controlled physical manipulations:
   - **Moved object** (known metric displacement measured with tape measure).
   - **Removed object** (present in *before*, absent in *after*, location observable in both).
   - **Added object** (absent in *before*, introduced into *after*, location observable in both).
   - **Static background** (structural walls and heavy furniture remaining stationary).

---

## 2. Capture Environment & Scene Setup

### 2.1 Room Selection
- **Recommended space**: Living room, bedroom, or office with dimensions approximately 3m × 3m to 6m × 6m.
- **Surface texture**: Ensure normal indoor texture (wood grain, books, rugs, posters). Avoid featureless white featureless walls or high-specular glass partitions.
- **Lighting**: Constant indoor illumination. Keep ceiling lights on; avoid moving window shadows or sudden fluorescent flicker.

### 2.2 Controlled Object Manipulations
Before recording, plan exactly 3 to 4 controlled changes and document them in the ground-truth table:
1. **Move #1 (Medium displacement)**:
   - Example: Shift an office chair by 1.20 meters across the floor.
   - Measure physical Euclidean distance with a tape measure from chair base center in *before* position to *after* position.
2. **Move #2 (Small displacement)**:
   - Example: Move a desk lamp, water bottle, or small box by 0.40–0.60 meters on a table.
3. **Removal #1**:
   - Example: Remove a backpack, coat, or pillow from a couch.
4. **Addition #1**:
   - Example: Place a cardboard box or plant pot onto an open floor area that was cleanly swept in the *before* video.
5. **Static Anchors**:
   - Desks, bookshelves, sofas, and room walls MUST remain untouched.

---

## 3. Smartphone Recording Guidelines

### 3.1 Camera Settings
- **Device**: Any modern iOS (iPhone 11+) or Android device (Galaxy S20+, Pixel 5+).
- **Resolution**: 1080p (1920×1080) or 4K at 30 fps (standard monocular color video; no LiDAR required).
- **Lens**: Standard wide lens (1x). Avoid ultra-wide (0.5x) due to severe uncalibrated radial barrel distortion.
- **Exposure / Focus**: Lock autofocus and auto-exposure (AE/AF lock) if possible to prevent sudden contrast shifts.

### 3.2 Trajectory & Motion
- **Video Duration**: 20–35 seconds per recording.
- **Pacing**: Walk slowly and smoothly. Avoid jerky wrist rotations or rapid panning (> 30°/second) to minimize motion blur.
- **Overlap**: Maintain 70–80% visual overlap between adjacent frames.
- **Trajectory Strategy**:
  - **Video 1 ("Before")**: Start at doorway, walk a smooth arc or semi-circle around the room clockwise, sweeping across the key furniture and objects.
  - **Intermission**: Pause, execute the planned object moves/removals/additions, verify static anchors did not move.
  - **Video 2 ("After")**: Start from a **different viewpoint** (e.g., opposite corner or counter-clockwise path), maintaining coverage of the same room objects and floor planes.

---

## 4. Ground-Truth Data Logging Format

Record ground-truth measurements in a JSON file matching RoomShift's evaluation format (`real_ground_truth.json`):

```json
[
  {
    "object_label": "chair",
    "change_type": "moved",
    "ground_truth_displacement_m": 1.25,
    "notes": "Office chair rolled from desk to window"
  },
  {
    "object_label": "bottle",
    "change_type": "moved",
    "ground_truth_displacement_m": 0.52,
    "notes": "Water bottle shifted to left side of desk"
  },
  {
    "object_label": "backpack",
    "change_type": "removed",
    "ground_truth_displacement_m": null,
    "notes": "Backpack on couch removed from room"
  },
  {
    "object_label": "box",
    "change_type": "added",
    "ground_truth_displacement_m": null,
    "notes": "Cardboard parcel placed on floor"
  }
]
```

---

## 5. Execution & Verification Workflow

### 5.1 CLI Execution
Run the end-to-end comparison on the captured videos:
```bash
python -m roomshift compare \
    --before /path/to/room_before.mp4 \
    --after /path/to/room_after.mp4 \
    --output outputs/real_room_evaluation \
    --verbose
```

### 5.2 Web UI Execution
Alternatively, launch the local dashboard:
```bash
python app.py
```
1. Open `http://localhost:5000` in a browser.
2. Drag and drop `room_before.mp4` and `room_after.mp4`.
3. Click **Detect Changes** and observe real-time pipeline stages.
4. Inspect the interactive 3D scene visualizer and download the structured JSON change report.

---

## 6. Honest Diagnostic Checks & Verification Criteria

1. **Reconstruction & Alignment Verification**:
   - Check `outputs/real_room_evaluation/change_report.json`.
   - Verify `registration_confidence`:
     - If $> 0.50$, alignment succeeded across camera trajectories.
     - If $< 0.30$, pipeline correctly marks alignment as untrusted and flags changes as `UNCERTAIN` rather than asserting false positives.
2. **Scale Ambiguity Honesty**:
   - Monocular camera pipelines possess an unobservable scale factor $s$. RoomShift AI explicitly outputs `scale_status: "relative"`.
   - Displacements are reported in relative reconstruction units unless metric reference calibration is provided.
3. **Visibility & Occlusion Honesty**:
   - If an object is not observed in the *after* recording (e.g. out of camera frame), RoomShift AI labels it `UNCERTAIN` with reason `"Object's original location was unobserved in the after recording"`. It does **not** falsely declare it `REMOVED`.
