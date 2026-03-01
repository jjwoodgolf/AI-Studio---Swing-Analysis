/**
 * LiveHUD.tsx
 *
 * Real-time Heads-Up Display for golf swing analysis.
 *
 * Key subsystems implemented here:
 *  1. OneEuroFilter  – adaptive low-pass filter (Casiez et al. 2012).
 *                      Kills jitter when the signal is slow while staying
 *                      responsive during fast movements.
 *  2. Depth-Normalised Angle – corrects for MediaPipe's z-scale ambiguity
 *                      so Shoulder Turn accuracy is preserved regardless of
 *                      the golfer's distance from the camera.
 *  3. LiveHUD component – overlays computed swing metrics on the video feed.
 */

import React, { useEffect, useRef, useState } from "react";

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

// ─── 4. HUD UI components ─────────────────────────────────────────────────────

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

// ─── 5. LiveHUD ───────────────────────────────────────────────────────────────

/**
 * `LiveHUD` overlays real-time golf swing metrics on top of a pose video feed.
 *
 * Usage:
 * ```tsx
 * <div style={{ position: "relative" }}>
 *   <video ref={videoRef} />
 *   <LiveHUD landmarks={poseLandmarks} onMetrics={handleMetrics} />
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
}) => {
  const [metrics, setMetrics] = useState<SwingMetrics | null>(null);

  // Each LiveHUD instance owns its own filter bank.
  const filtersRef = useRef<SwingFilters>(new SwingFilters());

  useEffect(() => {
    // Need a full 33-landmark array; bail early on partial / missing data.
    if (!landmarks || landmarks.length < 33) return;

    const m = computeMetrics(landmarks, filtersRef.current, performance.now());
    setMetrics(m);
    onMetrics?.(m);
  }, [landmarks, onMetrics]);

  // ── No-pose state ──────────────────────────────────────────────────────────
  if (!metrics) {
    return (
      <div
        className={className}
        style={rootStyle}
        aria-label="Swing HUD – awaiting pose"
      >
        <StatusDot confidence={0} />
        <span style={{ fontSize: 11, color: "#4a5568", marginLeft: 8 }}>
          Awaiting pose…
        </span>
      </div>
    );
  }

  const {
    shoulderTurn,
    shoulderTurnRaw,
    hipTurn,
    hipTurnRaw,
    xFactor,
    spineAngle,
    confidence,
  } = metrics;

  return (
    <div
      className={className}
      style={rootStyle}
      aria-label="Live Swing Analysis HUD"
    >
      {/* ── Header ────────────────────────────────────────────────────── */}
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
          Swing Analysis
        </span>
        <span style={{ marginLeft: "auto", fontSize: 10, color: "#4a5568" }}>
          {(confidence * 100).toFixed(0)}%
        </span>
      </div>

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
      <div
        style={{
          marginTop: 10,
          paddingTop: 8,
          borderTop: "1px solid rgba(255,255,255,0.05)",
          display: "flex",
          gap: 6,
        }}
      >
        <Badge>1€ Filter</Badge>
        <Badge>Depth Norm</Badge>
      </div>
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

const Badge: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span
    style={{
      fontSize: 8,
      color: "#4a5568",
      border: "1px solid rgba(255,255,255,0.06)",
      borderRadius: 4,
      padding: "1px 5px",
      textTransform: "uppercase",
      letterSpacing: "0.06em",
    }}
  >
    {children}
  </span>
);

// ─── Shared styles ────────────────────────────────────────────────────────────

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

export default LiveHUD;
