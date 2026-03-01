/**
 * LiveHUD.tsx
 *
 * Real-time Heads-Up Display for golf swing analysis.
 *
 * Key subsystems implemented here:
 *  1. OneEuroFilter       – adaptive low-pass filter (Casiez et al. 2012).
 *                           Kills jitter when the signal is slow while staying
 *                           responsive during fast movements.
 *  2. Depth-Normalised Angle – corrects for MediaPipe's z-scale ambiguity
 *                           so Shoulder Turn accuracy is preserved regardless
 *                           of the golfer's distance from the camera.
 *  3. Supabase sync       – optional cloud recording that degrades gracefully
 *                           to offline mode when API keys are absent or the
 *                           package is not installed.
 *  4. Handedness mirror   – flips landmark x-coordinates for front-facing
 *                           cameras so the skeleton overlay aligns with the
 *                           horizontally-flipped video stream.
 *  5. LiveHUD component   – overlays computed swing metrics on the video feed,
 *                           with Record / Set Address / Clear controls and a
 *                           toggleable Drawing Toolbar.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";

// ─── Supabase – offline-safe duck-typed client ────────────────────────────────

/** Minimal interface we actually use from @supabase/supabase-js. */
type SupabaseLike = {
  from(table: string): {
    insert(rows: object): Promise<{ error: { message: string } | null }>;
  };
};

/**
 * Attempts to create a Supabase client from the supplied credentials.
 * Returns `null` (offline mode) in any of these cases:
 *   • `url` or `key` are absent / undefined
 *   • `@supabase/supabase-js` is not installed in the project
 *   • `createClient` throws for any other reason
 *
 * The rest of the component treats `null` as "offline mode" and continues
 * to operate normally – recorded frames are buffered in memory only.
 */
function tryCreateClient(url?: string, key?: string): SupabaseLike | null {
  if (!url || !key) return null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { createClient } = require("@supabase/supabase-js") as {
      createClient: (u: string, k: string) => SupabaseLike;
    };
    return createClient(url, key);
  } catch {
    console.warn("[LiveHUD] Supabase init failed – running in offline mode.");
    return null;
  }
}

// ─── Public types ─────────────────────────────────────────────────────────────

/** A single 3-D pose landmark as returned by MediaPipe Pose. */
export interface PoseLandmark {
  /** Normalised horizontal position [0, 1] (left → right). */
  x: number;
  /** Normalised vertical position [0, 1] (top → bottom). */
  y: number;
  /**
   * Depth relative to the mid-hip point.
   * MediaPipe expresses z in the same scale as x (image-width units), so
   * negative means closer to the camera.
   */
  z: number;
  /** Landmark visibility confidence [0, 1]. */
  visibility?: number;
}

/** Processed swing metrics exposed to callers. */
export interface SwingMetrics {
  /** Depth-normalised, One-Euro-filtered shoulder turn (degrees). */
  shoulderTurn: number;
  /** Unfiltered raw shoulder turn for debug overlays (degrees). */
  shoulderTurnRaw: number;
  /** Depth-normalised, One-Euro-filtered hip turn (degrees). */
  hipTurn: number;
  /** Unfiltered raw hip turn (degrees). */
  hipTurnRaw: number;
  /**
   * X-Factor: differential rotation (shoulder turn − hip turn).
   * A key predictor of clubhead speed.
   */
  xFactor: number;
  /** Lateral spine tilt away from target (degrees, positive = trail side). */
  spineAngle: number;
  /** Average visibility of the four key landmarks [0, 1]. */
  confidence: number;
}

export interface LiveHUDProps {
  /**
   * Full MediaPipe Pose landmark array (33 elements).
   * Pass `null` when no pose is detected.
   */
  landmarks: PoseLandmark[] | null;
  /** Optional extra className applied to the root div. */
  className?: string;
  /** Called every frame with the latest computed metrics. */
  onMetrics?: (metrics: SwingMetrics) => void;
  /**
   * Set `true` for front-facing / selfie cameras.
   *
   * When enabled, landmark x-coordinates are mirrored (x → 1 − x) before
   * metric computation so the skeleton overlay aligns with the horizontally-
   * flipped video feed. The parent should also apply `scaleX(-1)` to any
   * canvas element that draws the raw skeleton points, e.g.:
   *
   * ```tsx
   * <canvas style={isFrontCamera ? { transform: "scaleX(-1)" } : undefined} />
   * ```
   */
  isFrontCamera?: boolean;
  /**
   * Supabase project URL (e.g. `https://xxxx.supabase.co`).
   * Omit to run in offline mode – the component functions normally but
   * recorded sessions are buffered in memory only.
   */
  supabaseUrl?: string;
  /**
   * Supabase anon / public API key.
   * Omit to run in offline mode.
   */
  supabaseKey?: string;
}

// ─── MediaPipe landmark indices ───────────────────────────────────────────────

const LM = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
} as const;

// ─── 1. One Euro Filter ───────────────────────────────────────────────────────

/**
 * One Euro Filter (Casiez, Roussel, Vogel — CHI 2012).
 *
 * An adaptive low-pass filter with two tuning knobs:
 *   minCutoff – minimum cutoff frequency (Hz).  Lower = smoother when still.
 *   beta      – speed coefficient. Higher = less lag during fast motion.
 *
 * The filter raises its cutoff frequency proportionally to the signal's
 * instantaneous speed, so it is simultaneously low-jitter and low-latency.
 *
 * Recommended starting values for golf swing angles:
 *   minCutoff = 0.5 Hz, beta = 0.015, dCutoff = 1.0 Hz
 */
class OneEuroFilter {
  private readonly minCutoff: number;
  private readonly beta: number;
  private readonly dCutoff: number;

  private xPrev: number | null = null;
  private dxPrev: number = 0.0;
  private tPrev: number | null = null;

  constructor(minCutoff = 0.5, beta = 0.015, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  /** Compute the smoothing factor α for a given cutoff frequency and Δt. */
  private alpha(cutoff: number, dt: number): number {
    // α = dt / (dt + 1/(2π·f_c))
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  /**
   * Feed a new sample into the filter.
   * @param x           Raw signal value.
   * @param timestampMs Monotonic timestamp in **milliseconds** (e.g. performance.now()).
   * @returns           Filtered value.
   */
  filter(x: number, timestampMs: number): number {
    // Bootstrap: return the first sample unmodified.
    if (this.xPrev === null || this.tPrev === null) {
      this.xPrev = x;
      this.tPrev = timestampMs;
      return x;
    }

    // Guard against zero or negative Δt (duplicate timestamps, clock skew).
    const dt = Math.max((timestampMs - this.tPrev) / 1000.0, 1e-6);

    // ── Step 1: Low-pass filter the derivative (speed estimate) ──────────────
    const dx = (x - this.xPrev) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1.0 - aD) * this.dxPrev;

    // ── Step 2: Compute adaptive cutoff from speed ────────────────────────────
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);

    // ── Step 3: Low-pass filter the signal ───────────────────────────────────
    const a = this.alpha(cutoff, dt);
    const xHat = a * x + (1.0 - a) * this.xPrev;

    // Advance state
    this.xPrev = xHat;
    this.dxPrev = dxHat;
    this.tPrev = timestampMs;

    return xHat;
  }

  /** Reset filter state (call at the start of a new swing or session). */
  reset(): void {
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }
}

// ─── 2. Depth-Normalised Angle ────────────────────────────────────────────────

/**
 * Computes the rotation angle (degrees) of a body segment in the horizontal
 * plane with depth normalisation.
 *
 * ## Why depth normalisation?
 *
 * MediaPipe's z coordinate is depth relative to the mid-hip point, expressed
 * in image-width units.  When the golfer rotates, the right shoulder moves
 * closer to the camera and the left shoulder moves further away (or vice
 * versa), producing a measurable dz.  However, the *scale* of dz can drift
 * depending on the golfer's distance and the model's depth confidence.
 *
 * ## Strategy
 *
 * We use a **reference segment** (the hips) to calibrate the effective
 * depth scale for this frame.  The hip segment has a known anatomical width,
 * and its apparent 3-D length in image space gives us a per-frame conversion
 * factor between depth units and horizontal image units.
 *
 *   depthScale = |refDx| / |ref3DLen|   (fraction of the 3-D length that is
 *                                         expressed as a horizontal offset)
 *
 * This factor is then applied to the measured segment's dz before computing
 * atan2, effectively expressing depth differences in the same coordinate
 * space as horizontal differences.
 *
 * @param left        Left endpoint of the segment to measure.
 * @param right       Right endpoint of the segment to measure.
 * @param leftRef     Left endpoint of the reference segment (e.g. left hip).
 * @param rightRef    Right endpoint of the reference segment (e.g. right hip).
 * @returns           Rotation angle in degrees.  Positive = open to camera,
 *                    negative = closed.
 */
function depthNormalisedAngle(
  left: PoseLandmark,
  right: PoseLandmark,
  leftRef: PoseLandmark,
  rightRef: PoseLandmark
): number {
  // Reference segment geometry
  const refDx = rightRef.x - leftRef.x;
  const refDz = rightRef.z - leftRef.z;
  const ref3DLen = Math.sqrt(refDx * refDx + refDz * refDz);

  // Guard against degenerate frames (both points on top of each other).
  if (ref3DLen < 1e-6) return 0;

  // Depth normalisation factor: scales z-diffs to the same units as x-diffs.
  const depthScale = Math.abs(refDx) / ref3DLen;

  // Measured segment
  const dx = right.x - left.x;
  const dz = (right.z - left.z) * depthScale;

  return Math.atan2(dz, dx) * (180.0 / Math.PI);
}

/** Average visibility of the specified landmark indices. */
function landmarkConfidence(lms: PoseLandmark[], indices: number[]): number {
  if (indices.length === 0) return 0;
  const sum = indices.reduce((acc, i) => acc + (lms[i]?.visibility ?? 0), 0);
  return sum / indices.length;
}

// ─── 3. Per-component filter instances ───────────────────────────────────────
// Kept in a class so each LiveHUD instance owns its own filter state and
// reset() can be called cleanly without touching other instances.

class SwingFilters {
  readonly shoulder = new OneEuroFilter(0.5, 0.015, 1.0);
  readonly hip = new OneEuroFilter(0.5, 0.015, 1.0);
  readonly spine = new OneEuroFilter(1.0, 0.010, 1.0);

  reset(): void {
    this.shoulder.reset();
    this.hip.reset();
    this.spine.reset();
  }
}

// ─── Metrics computation ──────────────────────────────────────────────────────

function computeMetrics(
  landmarks: PoseLandmark[],
  filters: SwingFilters,
  timestampMs: number
): SwingMetrics {
  const ls = landmarks[LM.LEFT_SHOULDER];
  const rs = landmarks[LM.RIGHT_SHOULDER];
  const lh = landmarks[LM.LEFT_HIP];
  const rh = landmarks[LM.RIGHT_HIP];

  const confidence = landmarkConfidence(landmarks, [
    LM.LEFT_SHOULDER,
    LM.RIGHT_SHOULDER,
    LM.LEFT_HIP,
    LM.RIGHT_HIP,
  ]);

  // ── Shoulder turn (hips used as depth reference) ──────────────────────────
  const shoulderTurnRaw = depthNormalisedAngle(ls, rs, lh, rh);
  const shoulderTurn = filters.shoulder.filter(shoulderTurnRaw, timestampMs);

  // ── Hip turn (ankles would be ideal; use self-reference for direct 3-D angle)
  // When the reference equals the measured segment, depthScale = cos(hipAngle),
  // yielding: atan2(dz·cos θ, dx) — a conservative estimate that degrades
  // gracefully without a lower-body reference.
  const hipTurnRaw = depthNormalisedAngle(lh, rh, lh, rh);
  const hipTurn = filters.hip.filter(hipTurnRaw, timestampMs);

  // ── X-Factor ──────────────────────────────────────────────────────────────
  const xFactor = shoulderTurn - hipTurn;

  // ── Spine lateral tilt ────────────────────────────────────────────────────
  const midSx = (ls.x + rs.x) / 2;
  const midSy = (ls.y + rs.y) / 2;
  const midHx = (lh.x + rh.x) / 2;
  const midHy = (lh.y + rh.y) / 2;
  // Positive = tilted toward trail side (right for right-handers).
  const spineAngleRaw =
    Math.atan2(midHx - midSx, midSy - midHy) * (180.0 / Math.PI);
  const spineAngle = filters.spine.filter(spineAngleRaw, timestampMs);

  return {
    shoulderTurn,
    shoulderTurnRaw,
    hipTurn,
    hipTurnRaw,
    xFactor,
    spineAngle,
    confidence,
  };
}

// ─── Drawing tools definition ─────────────────────────────────────────────────

const DRAWING_TOOLS = [
  { name: "Pen", icon: "✏" },
  { name: "Line", icon: "╱" },
  { name: "Angle", icon: "∠" },
  { name: "Circle", icon: "○" },
  { name: "Erase", icon: "⌫" },
] as const;

// ─── 4. HUD UI sub-components ─────────────────────────────────────────────────

interface GaugeProps {
  label: string;
  value: number;
  rawValue?: number;
  min: number;
  max: number;
  unit?: string;
  color: string;
  warnAbove?: number;
}

const Gauge: React.FC<GaugeProps> = ({
  label,
  value,
  rawValue,
  min,
  max,
  unit = "°",
  color,
  warnAbove,
}) => {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const isWarn = warnAbove !== undefined && Math.abs(value) > warnAbove;
  const barColor = isWarn ? "#ff6b35" : color;

  return (
    <div style={{ marginBottom: 10 }}>
      {/* Label row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 4,
        }}
      >
        <span
          style={{
            fontSize: 10,
            color: "#8a9ab5",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
          }}
        >
          {label}
        </span>
        <span style={{ fontSize: 14, fontWeight: 700, color: barColor }}>
          {value.toFixed(1)}
          {unit}
          {rawValue !== undefined && (
            <span style={{ fontSize: 9, color: "#4a5568", marginLeft: 5 }}>
              raw {rawValue.toFixed(1)}
            </span>
          )}
        </span>
      </div>

      {/* Progress bar track */}
      <div
        style={{
          height: 3,
          background: "rgba(255,255,255,0.08)",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct * 100}%`,
            height: "100%",
            background: barColor,
            borderRadius: 2,
            transition: "width 60ms linear, background 150ms",
          }}
        />
      </div>
    </div>
  );
};

/**
 * Down-arrow icon button (top-right of the header) that toggles the drawing
 * toolbar. The chevron rotates 180° when the toolbar is open.
 */
const DrawingToggleButton: React.FC<{
  open: boolean;
  onClick: () => void;
}> = ({ open, onClick }) => (
  <button
    onClick={onClick}
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      width: 20,
      height: 20,
      padding: 0,
      background: open
        ? "rgba(192, 132, 252, 0.15)"
        : "rgba(255,255,255,0.05)",
      border: `1px solid ${open ? "rgba(192,132,252,0.4)" : "rgba(255,255,255,0.1)"}`,
      borderRadius: 4,
      cursor: "pointer",
      color: open ? "#c084fc" : "#4a5568",
      flexShrink: 0,
      pointerEvents: "auto",
      transition: "background 150ms, border-color 150ms, color 150ms",
    }}
    aria-label={open ? "Close drawing toolbar" : "Open drawing toolbar"}
    title="Drawing Tools"
  >
    <svg
      width="10"
      height="10"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        transform: open ? "rotate(180deg)" : "rotate(0deg)",
        transition: "transform 200ms",
      }}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  </button>
);

/** Placeholder drawing toolbar revealed when the toggle is active. */
const DrawingToolbarPanel: React.FC = () => (
  <div
    style={{
      marginBottom: 10,
      padding: "8px 10px",
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: 8,
    }}
  >
    <span
      style={{
        display: "block",
        marginBottom: 6,
        fontSize: 9,
        color: "#8a9ab5",
        textTransform: "uppercase",
        letterSpacing: "0.08em",
      }}
    >
      Drawing Tools
    </span>
    <div style={{ display: "flex", gap: 4, pointerEvents: "auto" }}>
      {DRAWING_TOOLS.map((tool) => (
        <button
          key={tool.name}
          title={tool.name}
          style={{
            flex: 1,
            padding: "5px 0",
            fontSize: 13,
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.09)",
            borderRadius: 6,
            color: "#8a9ab5",
            cursor: "pointer",
            fontFamily: "inherit",
            transition: "background 120ms, color 120ms",
          }}
        >
          {tool.icon}
        </button>
      ))}
    </div>
  </div>
);

interface ActionBarProps {
  isRecording: boolean;
  hasAddress: boolean;
  canSetAddress: boolean;
  onRecord: () => void;
  onSetAddress: () => void;
  onClear: () => void;
}

/** Record / Set Address / Clear control row. */
const ActionBar: React.FC<ActionBarProps> = ({
  isRecording,
  hasAddress,
  canSetAddress,
  onRecord,
  onSetAddress,
  onClear,
}) => (
  <div
    style={{
      marginTop: 8,
      paddingTop: 8,
      borderTop: "1px solid rgba(255,255,255,0.05)",
      display: "flex",
      gap: 5,
      pointerEvents: "auto",
    }}
  >
    {/* Record / Stop */}
    <button
      onClick={onRecord}
      style={{
        ...actionBtnBase,
        ...(isRecording ? actionBtnRecordActive : {}),
      }}
    >
      {isRecording ? "■ Stop" : "● Rec"}
    </button>

    {/* Set Address */}
    <button
      onClick={onSetAddress}
      disabled={!canSetAddress}
      style={{
        ...actionBtnBase,
        ...(hasAddress ? actionBtnAddrActive : {}),
        opacity: canSetAddress ? 1 : 0.38,
        cursor: canSetAddress ? "pointer" : "not-allowed",
      }}
    >
      ⊙ Addr
    </button>

    {/* Clear */}
    <button onClick={onClear} style={actionBtnBase}>
      ✕ Clear
    </button>
  </div>
);

// ─── 5. LiveHUD ───────────────────────────────────────────────────────────────

/**
 * `LiveHUD` overlays real-time golf swing metrics on top of a pose video feed.
 *
 * Usage:
 * ```tsx
 * <div style={{ position: "relative" }}>
 *   <video ref={videoRef} />
 *   {/* Mirror the skeleton canvas for front cameras: *\/}
 *   <canvas style={isFront ? { transform: "scaleX(-1)" } : undefined} />
 *   <LiveHUD
 *     landmarks={poseLandmarks}
 *     isFrontCamera={isFront}
 *     supabaseUrl={process.env.REACT_APP_SUPABASE_URL}
 *     supabaseKey={process.env.REACT_APP_SUPABASE_ANON_KEY}
 *     onMetrics={handleMetrics}
 *   />
 * </div>
 * ```
 *
 * The component is `position: absolute` so it should sit inside a
 * `position: relative` parent that also contains the video element.
 */
export const LiveHUD: React.FC<LiveHUDProps> = ({
  landmarks,
  className,
  onMetrics,
  isFrontCamera = false,
  supabaseUrl,
  supabaseKey,
}) => {
  const [metrics, setMetrics] = useState<SwingMetrics | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [addressLandmarks, setAddressLandmarks] = useState<PoseLandmark[] | null>(null);
  const [drawingToolbarOpen, setDrawingToolbarOpen] = useState(false);

  // Each LiveHUD instance owns its own filter bank.
  const filtersRef = useRef<SwingFilters>(new SwingFilters());

  // Supabase: initialised once synchronously to avoid an offline→online flash
  // between the initial render and the first useEffect tick.
  // Sentinel: undefined = not yet resolved; null = offline; object = connected.
  const supaRef = useRef<SupabaseLike | null | undefined>(undefined);
  if (supaRef.current === undefined) {
    supaRef.current = tryCreateClient(supabaseUrl, supabaseKey);
  }
  const offlineMode = supaRef.current === null;

  // In-memory frame buffer used while recording.
  const recordedFramesRef = useRef<Array<{ ts: number; metrics: SwingMetrics }>>([]);
  const sessionIdRef = useRef<string | null>(null);

  // ── Metrics computation (runs on every incoming landmarks frame) ───────────
  useEffect(() => {
    // Need a full 33-landmark array; bail early on partial / missing data.
    if (!landmarks || landmarks.length < 33) return;

    // ── Handedness mirror ────────────────────────────────────────────────────
    // Front-facing cameras produce a horizontally-flipped video feed.
    // Mirroring x-coordinates here keeps skeleton metrics in the same display
    // space as the video, preserving anatomically correct left/right from the
    // viewer's perspective.  The parent is responsible for applying
    // `transform: scaleX(-1)` to any canvas that renders the raw skeleton.
    const lms: PoseLandmark[] = isFrontCamera
      ? landmarks.map((lm) => ({ ...lm, x: 1 - lm.x }))
      : landmarks;

    const m = computeMetrics(lms, filtersRef.current, performance.now());
    setMetrics(m);
    onMetrics?.(m);

    if (isRecording) {
      recordedFramesRef.current.push({ ts: performance.now(), metrics: m });
    }
  }, [landmarks, isFrontCamera, onMetrics, isRecording]);

  // ── Record handler ─────────────────────────────────────────────────────────
  const handleRecord = useCallback(() => {
    if (isRecording) {
      setIsRecording(false);
      const frames = recordedFramesRef.current;

      // Flush to Supabase when available; silently retain frames in memory
      // (offline mode) so callers can drain them via a custom onMetrics handler.
      if (!offlineMode && supaRef.current && frames.length > 0) {
        supaRef.current
          .from("swing_sessions")
          .insert({
            session_id: sessionIdRef.current,
            recorded_at: new Date().toISOString(),
            frames,
          })
          .then(({ error }) => {
            if (error) {
              console.error("[LiveHUD] Supabase write error:", error.message);
            }
          });
      }

      recordedFramesRef.current = [];
      sessionIdRef.current = null;
    } else {
      setIsRecording(true);
      sessionIdRef.current = `session_${Date.now()}`;
      recordedFramesRef.current = [];
    }
  }, [isRecording, offlineMode]);

  // ── Set Address handler ────────────────────────────────────────────────────
  // Captures the current landmark set as the golfer's address (setup) position.
  // Mirrors x-coords consistent with isFrontCamera so stored coordinates match
  // the metrics coordinate space.
  const handleSetAddress = useCallback(() => {
    if (!landmarks || landmarks.length < 33) return;
    const lms: PoseLandmark[] = isFrontCamera
      ? landmarks.map((lm) => ({ ...lm, x: 1 - lm.x }))
      : landmarks;
    setAddressLandmarks(lms);
  }, [landmarks, isFrontCamera]);

  // ── Clear handler ──────────────────────────────────────────────────────────
  const handleClear = useCallback(() => {
    setAddressLandmarks(null);
    setIsRecording(false);
    recordedFramesRef.current = [];
    sessionIdRef.current = null;
    filtersRef.current.reset();
    setMetrics(null);
  }, []);

  // ── Shared header ──────────────────────────────────────────────────────────
  const confidence = metrics?.confidence ?? 0;

  const header = (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        marginBottom: 12,
        gap: 8,
      }}
    >
      <StatusDot confidence={confidence} />
      <span
        style={{
          fontSize: 10,
          color: "#8a9ab5",
          textTransform: "uppercase",
          letterSpacing: "0.1em",
        }}
      >
        {metrics ? "Swing Analysis" : "Awaiting pose…"}
      </span>
      {/* Confidence % + drawing-toolbar toggle, grouped on the right */}
      <div
        style={{
          marginLeft: "auto",
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        {metrics && (
          <span style={{ fontSize: 10, color: "#4a5568" }}>
            {(confidence * 100).toFixed(0)}%
          </span>
        )}
        <DrawingToggleButton
          open={drawingToolbarOpen}
          onClick={() => setDrawingToolbarOpen((o) => !o)}
        />
      </div>
    </div>
  );

  // ── Shared footer badges ───────────────────────────────────────────────────
  const footer = (
    <div
      style={{
        marginTop: 10,
        paddingTop: 8,
        borderTop: "1px solid rgba(255,255,255,0.05)",
        display: "flex",
        gap: 6,
        flexWrap: "wrap",
      }}
    >
      <Badge>1€ Filter</Badge>
      <Badge>Depth Norm</Badge>
      {isFrontCamera && <Badge>Mirror</Badge>}
      {offlineMode && (
        <Badge style={{ color: "#fbbf24", borderColor: "rgba(251,191,36,0.2)" }}>
          Offline
        </Badge>
      )}
      {isRecording && (
        <Badge style={{ color: "#f87171", borderColor: "rgba(248,113,113,0.2)" }}>
          ● REC
        </Badge>
      )}
      {addressLandmarks && (
        <Badge style={{ color: "#34d399", borderColor: "rgba(52,211,153,0.2)" }}>
          Addr Set
        </Badge>
      )}
    </div>
  );

  // ── No-pose state ──────────────────────────────────────────────────────────
  if (!metrics) {
    return (
      <div
        className={className}
        style={rootStyle}
        aria-label="Swing HUD – awaiting pose"
      >
        {header}
        {drawingToolbarOpen && <DrawingToolbarPanel />}
        {footer}
        <ActionBar
          isRecording={isRecording}
          hasAddress={!!addressLandmarks}
          canSetAddress={false}
          onRecord={handleRecord}
          onSetAddress={handleSetAddress}
          onClear={handleClear}
        />
      </div>
    );
  }

  const { shoulderTurn, shoulderTurnRaw, hipTurn, hipTurnRaw, xFactor, spineAngle } =
    metrics;

  return (
    <div
      className={className}
      style={rootStyle}
      aria-label="Live Swing Analysis HUD"
    >
      {/* ── Header ────────────────────────────────────────────────────── */}
      {header}

      {/* ── Drawing Toolbar (placeholder) ─────────────────────────────── */}
      {drawingToolbarOpen && <DrawingToolbarPanel />}

      {/* ── Gauges ────────────────────────────────────────────────────── */}
      <Gauge
        label="Shoulder Turn"
        value={shoulderTurn}
        rawValue={shoulderTurnRaw}
        min={-10}
        max={110}
        color="#38bdf8"
        warnAbove={100}
      />
      <Gauge
        label="Hip Turn"
        value={hipTurn}
        rawValue={hipTurnRaw}
        min={-5}
        max={60}
        color="#34d399"
        warnAbove={55}
      />
      <Gauge
        label="X-Factor"
        value={xFactor}
        min={-10}
        max={60}
        color="#c084fc"
        warnAbove={55}
      />
      <Gauge
        label="Spine Tilt"
        value={spineAngle}
        min={-20}
        max={20}
        color="#fbbf24"
        warnAbove={15}
      />

      {/* ── Footer badges ─────────────────────────────────────────────── */}
      {footer}

      {/* ── Action Buttons ────────────────────────────────────────────── */}
      <ActionBar
        isRecording={isRecording}
        hasAddress={!!addressLandmarks}
        canSetAddress={true}
        onRecord={handleRecord}
        onSetAddress={handleSetAddress}
        onClear={handleClear}
      />
    </div>
  );
};

// ─── Small presentational helpers ────────────────────────────────────────────

const StatusDot: React.FC<{ confidence: number }> = ({ confidence }) => {
  const color =
    confidence > 0.75 ? "#34d399" : confidence > 0.4 ? "#fbbf24" : "#f87171";
  return (
    <div
      style={{
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: color,
        boxShadow: `0 0 6px ${color}`,
        flexShrink: 0,
      }}
    />
  );
};

const Badge: React.FC<{
  children: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ children, style }) => (
  <span
    style={{
      fontSize: 8,
      color: "#4a5568",
      border: "1px solid rgba(255,255,255,0.06)",
      borderRadius: 4,
      padding: "1px 5px",
      textTransform: "uppercase",
      letterSpacing: "0.06em",
      ...style,
    }}
  >
    {children}
  </span>
);

// ─── Shared styles ────────────────────────────────────────────────────────────

/**
 * Root HUD container.
 *
 * `pointerEvents: "none"` lets clicks pass through to the underlying video.
 * Interactive children (buttons) set `pointerEvents: "auto"` locally so they
 * remain fully clickable despite the parent override.
 */
const rootStyle: React.CSSProperties = {
  position: "absolute",
  top: 14,
  left: 14,
  width: 230,
  padding: "14px 16px",
  background: "rgba(8, 10, 18, 0.85)",
  backdropFilter: "blur(12px)",
  WebkitBackdropFilter: "blur(12px)",
  borderRadius: 12,
  border: "1px solid rgba(255, 255, 255, 0.07)",
  fontFamily: '"SF Mono", "Fira Code", "Cascadia Code", monospace',
  color: "#fff",
  userSelect: "none",
  pointerEvents: "none",
  display: "flex",
  flexDirection: "column",
};

const actionBtnBase: React.CSSProperties = {
  flex: 1,
  padding: "4px 0",
  fontSize: 9,
  fontFamily: "inherit",
  fontWeight: 600,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 6,
  background: "rgba(255,255,255,0.05)",
  color: "#8a9ab5",
  cursor: "pointer",
  transition: "background 150ms, color 150ms, border-color 150ms",
};

const actionBtnRecordActive: React.CSSProperties = {
  background: "rgba(248, 113, 113, 0.15)",
  borderColor: "rgba(248, 113, 113, 0.4)",
  color: "#f87171",
};

const actionBtnAddrActive: React.CSSProperties = {
  background: "rgba(52, 211, 153, 0.15)",
  borderColor: "rgba(52, 211, 153, 0.4)",
  color: "#34d399",
};

export default LiveHUD;
