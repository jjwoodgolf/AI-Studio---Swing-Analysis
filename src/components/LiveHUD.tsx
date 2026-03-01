import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, PoseLandmarker, Landmark, DrawingUtils } from '@mediapipe/tasks-vision';
import { Camera, RefreshCw, Settings2, AlertTriangle, CheckCircle2, Upload, Play, Pause } from 'lucide-react';
import { supabase } from '../lib/supabase';

const METER_TO_INCH = 39.3701;
const METER_TO_CM = 100;
const EMA_ALPHA = 0.08;
const HUD_UPDATE_INTERVAL_MS = 250; // 4 times per second
const Z_STABILIZATION_THRESHOLD = 0.01;
const STILLNESS_THRESHOLD = 0.005; // 5mm noise gate
const ADDRESS_LOCK_RADIUS_M = 0.0254; // 1 inch in meters
const METRIC_SMOOTHING_ALPHA = 0.3; // Hysteresis smoothing

type MetricId = 'shoulderTilt' | 'shoulderTurn' | 'hipTurn' | 'leadArmBend' | 'forwardBend' | 'sway' | 'lift' | 'thrust' | 'xFactor';
type CameraView = 'Face-On' | 'Down-the-Line';
type Unit = 'in' | 'cm';

interface MetricDef {
  id: MetricId;
  label: string;
  type: 'angle' | 'linear';
  requiredView: CameraView | 'Any';
}

const AVAILABLE_METRICS: MetricDef[] = [
  { id: 'shoulderTilt', label: 'Shoulder Tilt', type: 'angle', requiredView: 'Face-On' },
  { id: 'shoulderTurn', label: 'Shoulder Turn', type: 'angle', requiredView: 'Down-the-Line' },
  { id: 'hipTurn', label: 'Hip Turn', type: 'angle', requiredView: 'Down-the-Line' },
  { id: 'xFactor', label: 'X-Factor', type: 'angle', requiredView: 'Down-the-Line' },
  { id: 'leadArmBend', label: 'Lead Arm Bend', type: 'angle', requiredView: 'Any' },
  { id: 'forwardBend', label: 'Spine Angle', type: 'angle', requiredView: 'Down-the-Line' },
  { id: 'sway', label: 'Hip Sway', type: 'linear', requiredView: 'Face-On' },
  { id: 'lift', label: 'Lift', type: 'linear', requiredView: 'Face-On' },
  { id: 'thrust', label: 'Thrust', type: 'linear', requiredView: 'Down-the-Line' },
];

export default function LiveHUD() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [isLoaded, setIsLoaded] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isVideoFile, setIsVideoFile] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [cameraError, setCameraError] = useState(false);

  const [videoProgress, setVideoProgress] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  // Settings State
  const [selectedMetrics, setSelectedMetrics] = useState<MetricId[]>(['shoulderTurn', 'shoulderTilt', 'forwardBend']);
  const [cameraView, setCameraView] = useState<CameraView>('Face-On');
  const [linearUnit, setLinearUnit] = useState<Unit>('in');
  const [showSettings, setShowSettings] = useState(false);

  // Tracking State
  const [metricsValues, setMetricsValues] = useState<Record<MetricId, number>>({} as Record<MetricId, number>);
  const [addressState, setAddressState] = useState<{ x: number; y: number; z: number } | null>(null);
  const [addressAngles, setAddressAngles] = useState<Record<MetricId, number> | null>(null);
  const [addressFeedback, setAddressFeedback] = useState(false);
  const [landmarksDetected, setLandmarksDetected] = useState(false);
  const [activeHighlights, setActiveHighlights] = useState<Record<string, boolean>>({});

  // AI Coach State
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<'idle' | 'processing' | 'completed' | 'error'>('idle');
  const [aiTip, setAiTip] = useState<string | null>(null);

  const requestRef = useRef<number>();
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const smoothedLandmarksRef = useRef<Landmark[] | null>(null);
  const smoothedDrawLandmarksRef = useRef<{ x: number; y: number; z?: number; visibility?: number }[] | null>(null);
  const lastUpdateTimeRef = useRef<number>(0);
  const highlightTimeoutRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const metricsValuesRef = useRef<Record<MetricId, number>>({} as Record<MetricId, number>);

  useEffect(() => {
    let active = true;
    const initializeMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1,
          outputSegmentationMasks: false,
        });
        if (active) {
          landmarkerRef.current = landmarker;
          setIsLoaded(true);
        }
      } catch (error) {
        console.error("Error initializing MediaPipe:", error);
      }
    };
    initializeMediaPipe();
    return () => {
      active = false;
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (landmarkerRef.current) landmarkerRef.current.close();
    };
  }, []);

  useEffect(() => {
    if (!sessionId || !supabase) return;

    const channel = supabase
      .channel(`swing_sessions_${sessionId}`)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'swing_sessions',
          filter: `id=eq.${sessionId}`,
        },
        (payload) => {
          const newStatus = payload.new.status as string;
          setAnalysisStatus(newStatus as 'idle' | 'processing' | 'completed' | 'error');
          if (newStatus === 'completed') {
            setAiTip(payload.new.ai_tip as string);
          }
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [sessionId]);

  const [isFrontCamera, setIsFrontCamera] = useState(true);

  const startCamera = async () => {
    if (!videoRef.current) return;
    try {
      setCameraError(false);
      const constraints = {
        video: {
          width: { ideal: 720 },
          height: { ideal: 1280 },
          facingMode: "user"
        }
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      setIsFrontCamera(settings.facingMode !== 'environment');

      videoRef.current.srcObject = stream;
      videoRef.current.play();
      setIsCameraActive(true);
      setIsVideoFile(false);
    } catch (err) {
      console.error("Error accessing camera:", err);
      setCameraError(true);
    }
  };

  const triggerAnalysis = async () => {
    if (!supabase) {
      console.warn("Supabase not configured. Skipping analysis.");
      return;
    }

    setAnalysisStatus('processing');
    setAiTip(null);

    const { data, error } = await supabase
      .from('swing_sessions')
      .insert([
        { status: 'processing', metrics_data: metricsValuesRef.current }
      ])
      .select()
      .single();

    if (error) {
      console.error("Error inserting session:", error);
      setAnalysisStatus('error');
      return;
    }

    setSessionId(data.id);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !videoRef.current) return;

    const url = URL.createObjectURL(file);
    videoRef.current.srcObject = null;
    videoRef.current.src = url;
    videoRef.current.play();
    setIsCameraActive(true);
    setIsVideoFile(true);
    setIsPlaying(true);
    setCameraError(false);

    videoRef.current.onloadedmetadata = () => {
      setVideoDuration(videoRef.current?.duration || 0);
    };

    triggerAnalysis();
  };

  const stopCamera = () => {
    if (videoRef.current) {
      if (videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
      videoRef.current.srcObject = null;
      videoRef.current.src = '';
      setIsCameraActive(false);
      setIsVideoFile(false);
      setIsRecording(false);
      setIsPlaying(false);
      smoothedLandmarksRef.current = null;
      setAnalysisStatus('idle');
      setAiTip(null);
    }
  };

  const toggleMetric = (id: MetricId) => {
    if (selectedMetrics.includes(id)) {
      setSelectedMetrics(selectedMetrics.filter(m => m !== id));
    } else if (selectedMetrics.length < 3) {
      setSelectedMetrics([...selectedMetrics, id]);
    }
  };

  // Math Helpers
  const applyEMA = (current: Landmark[], previous: Landmark[] | null): Landmark[] => {
    if (!previous) return current;
    return current.map((pt, i) => {
      const prev = previous[i];
      let newZ = pt.z;
      if (Math.abs(pt.z - prev.z) < Math.abs(prev.z * Z_STABILIZATION_THRESHOLD)) {
        newZ = prev.z;
      }
      return {
        x: prev.x * (1 - EMA_ALPHA) + pt.x * EMA_ALPHA,
        y: prev.y * (1 - EMA_ALPHA) + pt.y * EMA_ALPHA,
        z: prev.z * (1 - EMA_ALPHA) + newZ * EMA_ALPHA,
        visibility: pt.visibility,
      };
    });
  };

  const applyEMA2D = (
    current: { x: number; y: number; z?: number; visibility?: number }[],
    previous: { x: number; y: number; z?: number; visibility?: number }[] | null
  ): { x: number; y: number; z?: number; visibility?: number }[] => {
    if (!previous) return current;
    return current.map((pt, i) => {
      const prev = previous[i];
      let newZ = pt.z;
      if (pt.z !== undefined && prev.z !== undefined && Math.abs(pt.z - prev.z) < Math.abs(prev.z * Z_STABILIZATION_THRESHOLD)) {
        newZ = prev.z;
      }
      return {
        x: prev.x * (1 - EMA_ALPHA) + pt.x * EMA_ALPHA,
        y: prev.y * (1 - EMA_ALPHA) + pt.y * EMA_ALPHA,
        z: prev.z !== undefined ? (prev.z * (1 - EMA_ALPHA) + (newZ ?? 0) * EMA_ALPHA) : undefined,
        visibility: pt.visibility,
      };
    });
  };

  const calculateTilt = (a: Landmark, b: Landmark) => {
    const dy = b.y - a.y;
    const dx = b.x - a.x;
    let angle = Math.abs((Math.atan2(dy, dx) * 180) / Math.PI);
    if (angle > 90) angle = 180 - angle;
    return angle;
  };

  const calculateTurn = (left: Landmark, right: Landmark) => {
    const dz = right.z - left.z;
    const dx = right.x - left.x;
    return (Math.atan2(dz, dx) * 180) / Math.PI;
  };

  const calculateSpineAngle = (shoulder: Landmark, hip: Landmark) => {
    const dy = hip.y - shoulder.y;
    const dx = hip.x - shoulder.x;
    const dz = hip.z - shoulder.z;
    const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return (Math.asin(Math.abs(dy) / length) * 180) / Math.PI;
  };

  const calculateAngle3D = (a: Landmark, b: Landmark, c: Landmark) => {
    const v1 = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
    const v2 = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
    const dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    return (Math.acos(dotProduct / (mag1 * mag2)) * 180) / Math.PI;
  };

  const processVideo = () => {
    if (!videoRef.current || !landmarkerRef.current || !isCameraActive) {
      requestRef.current = requestAnimationFrame(processVideo);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.readyState >= 2) {
      const startTimeMs = performance.now();
      const results = landmarkerRef.current.detectForVideo(video, startTimeMs);

      const hasLandmarks = results.landmarks && results.landmarks.length > 0;
      setLandmarksDetected(hasLandmarks);

      // Canvas Drawing
      if (canvas) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        const canvasCtx = canvas.getContext('2d');
        if (canvasCtx) {
          canvasCtx.save();
          canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

          if (hasLandmarks) {
            const drawingUtils = new DrawingUtils(canvasCtx);
            const rawDrawLandmarks = results.landmarks[0];
            const smoothedDrawLandmarks = applyEMA2D(rawDrawLandmarks, smoothedDrawLandmarksRef.current);
            smoothedDrawLandmarksRef.current = smoothedDrawLandmarks;

            drawingUtils.drawConnectors(smoothedDrawLandmarks, PoseLandmarker.POSE_CONNECTIONS, {
              color: 'rgba(255, 255, 255, 0.5)',
              lineWidth: 2
            });
            drawingUtils.drawLandmarks(smoothedDrawLandmarks, {
              color: '#10b981',
              lineWidth: 2,
              radius: 3
            });
          }
          canvasCtx.restore();
        }
      }

      if (results.worldLandmarks && results.worldLandmarks.length > 0) {
        let rawLandmarks = results.worldLandmarks[0];

        // Horizon Calibration
        const leftAnkle = rawLandmarks[27];
        const rightAnkle = rawLandmarks[28];
        if (leftAnkle && rightAnkle) {
          const horizonAngle = Math.atan2(rightAnkle.y - leftAnkle.y, rightAnkle.x - leftAnkle.x);
          const cosA = Math.cos(-horizonAngle);
          const sinA = Math.sin(-horizonAngle);

          rawLandmarks = rawLandmarks.map(pt => ({
            ...pt,
            x: pt.x * cosA - pt.y * sinA,
            y: pt.x * sinA + pt.y * cosA,
          }));
        }

        // Velocity Check (Stillness)
        let isStill = false;
        if (smoothedLandmarksRef.current) {
          let totalDelta = 0;
          for (let i = 0; i < rawLandmarks.length; i++) {
            const dx = rawLandmarks[i].x - smoothedLandmarksRef.current[i].x;
            const dy = rawLandmarks[i].y - smoothedLandmarksRef.current[i].y;
            const dz = rawLandmarks[i].z - smoothedLandmarksRef.current[i].z;
            totalDelta += Math.sqrt(dx * dx + dy * dy + dz * dz);
          }
          if ((totalDelta / rawLandmarks.length) < STILLNESS_THRESHOLD) {
            isStill = true;
          }
        }

        const smoothedLandmarks = applyEMA(rawLandmarks, smoothedLandmarksRef.current);
        smoothedLandmarksRef.current = smoothedLandmarks;

        // Throttle HUD updates
        if (startTimeMs - lastUpdateTimeRef.current >= HUD_UPDATE_INTERVAL_MS && !isStill) {
          lastUpdateTimeRef.current = startTimeMs;

          const leftShoulder = smoothedLandmarks[11];
          const rightShoulder = smoothedLandmarks[12];
          const leftElbow = smoothedLandmarks[13];
          const leftWrist = smoothedLandmarks[15];
          const leftHip = smoothedLandmarks[23];
          const rightHip = smoothedLandmarks[24];

          const midShoulder: Landmark = {
            x: (leftShoulder.x + rightShoulder.x) / 2,
            y: (leftShoulder.y + rightShoulder.y) / 2,
            z: (leftShoulder.z + rightShoulder.z) / 2,
            visibility: 1
          };

          const midHip: Landmark = {
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2,
            z: (leftHip.z + rightHip.z) / 2,
            visibility: 1
          };

          const multiplier = linearUnit === 'in' ? METER_TO_INCH : METER_TO_CM;

          let swayMeters = addressState ? Math.abs(midHip.x - addressState.x) : 0;
          let liftMeters = addressState ? Math.abs(midHip.y - addressState.y) : 0;
          let thrustMeters = addressState ? Math.abs(midHip.z - addressState.z) : 0;

          // Address Position Locking (1-inch radius)
          if (swayMeters < ADDRESS_LOCK_RADIUS_M) swayMeters = 0;
          if (liftMeters < ADDRESS_LOCK_RADIUS_M) liftMeters = 0;
          if (thrustMeters < ADDRESS_LOCK_RADIUS_M) thrustMeters = 0;

          const rawTargetValues: Record<MetricId, number> = {
            shoulderTilt: calculateTilt(leftShoulder, rightShoulder),
            shoulderTurn: calculateTurn(leftShoulder, rightShoulder),
            hipTurn: calculateTurn(leftHip, rightHip),
            leadArmBend: calculateAngle3D(leftShoulder, leftElbow, leftWrist),
            forwardBend: calculateSpineAngle(midShoulder, midHip),
            sway: swayMeters * multiplier,
            lift: liftMeters * multiplier,
            thrust: thrustMeters * multiplier,
            xFactor: 0, // Calculated after baseline subtraction
          };

          const targetValues: Record<MetricId, number> = { ...rawTargetValues };

          // X-Factor is the difference between absolute shoulder turn and hip turn
          targetValues.xFactor = Math.max(0, Math.abs(targetValues.shoulderTurn) - Math.abs(targetValues.hipTurn));

          const newValues: Record<MetricId, number> = {} as Record<MetricId, number>;
          for (const key of Object.keys(targetValues) as MetricId[]) {
            let target = targetValues[key];
            const prev = metricsValuesRef.current[key];

            if (prev !== undefined) {
              // 0.5-degree precision gate for angles
              const def = AVAILABLE_METRICS.find(m => m.id === key);
              if (def?.type === 'angle' && Math.abs(target - prev) < 0.5) {
                target = prev;
              }
              // Hysteresis smoothing: Ease-In transition
              const alpha = (key === 'shoulderTurn' || key === 'forwardBend') ? 0.15 : METRIC_SMOOTHING_ALPHA;
              newValues[key] = prev + (target - prev) * alpha;
            } else {
              newValues[key] = target;
            }
          }
          metricsValuesRef.current = newValues;

          // Highlight Logic
          if (newValues.shoulderTurn >= 90) {
            setActiveHighlights(prev => ({ ...prev, shoulderTurn: true }));
            if (highlightTimeoutRef.current['shoulderTurn']) clearTimeout(highlightTimeoutRef.current['shoulderTurn']);
            highlightTimeoutRef.current['shoulderTurn'] = setTimeout(() => {
              setActiveHighlights(prev => ({ ...prev, shoulderTurn: false }));
            }, 500);
          }

          setMetricsValues(newValues);
        }
      }
    }

    requestRef.current = requestAnimationFrame(processVideo);
  };

  useEffect(() => {
    if (isCameraActive && isLoaded) {
      requestRef.current = requestAnimationFrame(processVideo);
    } else if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isCameraActive, isLoaded, addressState, addressAngles, linearUnit]);

  const calibrateAddress = () => {
    if (smoothedLandmarksRef.current) {
      const leftShoulder = smoothedLandmarksRef.current[11];
      const rightShoulder = smoothedLandmarksRef.current[12];
      const leftElbow = smoothedLandmarksRef.current[13];
      const leftWrist = smoothedLandmarksRef.current[15];
      const leftHip = smoothedLandmarksRef.current[23];
      const rightHip = smoothedLandmarksRef.current[24];

      const midShoulder: Landmark = {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2,
        z: (leftShoulder.z + rightShoulder.z) / 2,
        visibility: 1
      };

      const midHip: Landmark = {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2,
        z: (leftHip.z + rightHip.z) / 2,
        visibility: 1
      };

      setAddressState({
        x: midHip.x,
        y: midHip.y,
        z: midHip.z,
      });

      setAddressAngles({
        shoulderTilt: calculateTilt(leftShoulder, rightShoulder),
        shoulderTurn: calculateTurn(leftShoulder, rightShoulder),
        hipTurn: calculateTurn(leftHip, rightHip),
        leadArmBend: calculateAngle3D(leftShoulder, leftElbow, leftWrist),
        forwardBend: calculateSpineAngle(midShoulder, midHip),
        sway: 0,
        lift: 0,
        thrust: 0,
        xFactor: 0,
      });

      setAddressFeedback(true);
      setTimeout(() => setAddressFeedback(false), 2000);
    }
  };

  const clearAddress = () => {
    setAddressState(null);
    setAddressAngles(null);
  };

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setVideoProgress(time);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  };

  return (
    <div className="relative w-full max-w-md mx-auto bg-zinc-900 rounded-3xl overflow-hidden border border-zinc-800 shadow-2xl">

      {/* Settings Panel */}
      {showSettings && (
        <div className="absolute inset-0 z-50 bg-zinc-950/95 backdrop-blur-md p-6 overflow-y-auto">
          <div className="flex justify-between items-center mb-8">
            <h3 className="text-2xl font-bold text-white">HUD Settings</h3>
            <button onClick={() => setShowSettings(false)} className="text-zinc-400 hover:text-white">Close</button>
          </div>

          <div className="space-y-8">
            {/* Rule of 3 Selector */}
            <div className="space-y-4">
              <div className="flex justify-between items-end">
                <h4 className="text-lg font-semibold text-emerald-400">Select Metrics (Rule of 3)</h4>
                <span className="text-sm text-zinc-400">{selectedMetrics.length}/3 Selected</span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {AVAILABLE_METRICS.map(metric => {
                  const isSelected = selectedMetrics.includes(metric.id);
                  const isDisabled = !isSelected && selectedMetrics.length >= 3;
                  return (
                    <button
                      key={metric.id}
                      onClick={() => toggleMetric(metric.id)}
                      disabled={isDisabled}
                      className={`p-3 rounded-xl border text-left transition-all ${
                        isSelected
                          ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400'
                          : isDisabled
                            ? 'bg-zinc-900/50 border-zinc-800 text-zinc-600 cursor-not-allowed'
                            : 'bg-zinc-900 border-zinc-700 text-zinc-300 hover:border-zinc-500'
                      }`}
                    >
                      <div className="font-medium text-sm">{metric.label}</div>
                      <div className="text-[10px] opacity-70 mt-1">{metric.type === 'angle' ? 'Degrees' : 'Linear'}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Camera View & Units */}
            <div className="grid grid-cols-1 gap-6">
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Camera View</h4>
                <div className="flex bg-zinc-900 p-1 rounded-lg border border-zinc-800">
                  {(['Face-On', 'Down-the-Line'] as CameraView[]).map(view => (
                    <button
                      key={view}
                      onClick={() => setCameraView(view)}
                      className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${
                        cameraView === view ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'
                      }`}
                    >
                      {view}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Linear Units</h4>
                <div className="flex bg-zinc-900 p-1 rounded-lg border border-zinc-800">
                  {(['in', 'cm'] as Unit[]).map(unit => (
                    <button
                      key={unit}
                      onClick={() => setLinearUnit(unit)}
                      className={`flex-1 py-2 text-sm font-medium rounded-md transition-all uppercase ${
                        linearUnit === unit ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'
                      }`}
                    >
                      {unit === 'in' ? 'Inches' : 'Centimeters'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Camera Viewport */}
      <div className="relative aspect-[9/16] bg-black">
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: isVideoFile ? 'none' : (isFrontCamera ? 'scaleX(-1)' : 'none') }}
          playsInline
          muted
          loop={isVideoFile}
          onTimeUpdate={() => {
            if (videoRef.current) {
              setVideoProgress(videoRef.current.currentTime);
            }
          }}
          onEnded={() => setIsPlaying(false)}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover z-10 pointer-events-none"
          style={{ transform: isVideoFile ? 'none' : (isFrontCamera ? 'scaleX(-1)' : 'none') }}
        />

        {/* AI Coach Banner */}
        {analysisStatus !== 'idle' && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 w-[90%] max-w-sm">
            <div className={`backdrop-blur-md rounded-2xl p-4 border shadow-2xl transition-all duration-500 ${
              analysisStatus === 'processing' ? 'bg-zinc-900/90 border-emerald-500/50' :
              analysisStatus === 'completed' ? 'bg-emerald-950/90 border-emerald-500' :
              'bg-red-950/90 border-red-500'
            }`}>
              {analysisStatus === 'processing' && (
                <div className="flex items-center space-x-3 text-emerald-400">
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span className="font-medium">Coach is thinking...</span>
                </div>
              )}
              {analysisStatus === 'completed' && (
                <div className="space-y-1">
                  <div className="flex items-center space-x-2 text-emerald-400 mb-2">
                    <CheckCircle2 className="w-5 h-5" />
                    <span className="font-bold text-sm uppercase tracking-wider">Casual Tip</span>
                  </div>
                  <p className="text-white text-sm leading-relaxed">{aiTip}</p>
                </div>
              )}
              {analysisStatus === 'error' && (
                <div className="flex items-center space-x-3 text-red-400">
                  <AlertTriangle className="w-5 h-5" />
                  <span className="font-medium">Analysis failed. Try another angle.</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Visual Alignment HUD */}
        {isCameraActive && !isRecording && !isVideoFile && (
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center" style={{ zIndex: 15 }}>
            {/* Vertical Center Line */}
            <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-emerald-400/50" />
            {/* Horizontal Horizon Line */}
            <div className="absolute left-0 right-0 top-1/2 h-[1px] bg-emerald-400/50" />
            {/* Instruction */}
            <div className="absolute top-1/4 bg-zinc-950/80 backdrop-blur-sm text-emerald-400 text-xs sm:text-sm px-4 py-2 rounded-full border border-emerald-500/30 text-center max-w-[80%]">
              Align vertical line with target path and horizontal line with the ground.
            </div>
          </div>
        )}

        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/80 backdrop-blur-sm z-10">
            <Camera className="w-16 h-16 text-zinc-600 mb-4" />
            <p className="text-zinc-400 mb-6 text-center px-4">
              {cameraError
                ? "Camera inactive (PC) - Use 'Upload Swing' to test."
                : "Camera is inactive"}
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <button
                onClick={startCamera}
                disabled={!isLoaded}
                className="bg-emerald-500 hover:bg-emerald-400 text-zinc-950 font-semibold py-3 px-8 rounded-full transition-all disabled:opacity-50 flex items-center justify-center space-x-2 shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)]"
              >
                {isLoaded ? (
                  <>
                    <Camera className="w-5 h-5" />
                    <span>Start Live Tracking</span>
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    <span>Loading AI Models...</span>
                  </>
                )}
              </button>

              <label className={`cursor-pointer border border-emerald-400 text-emerald-400 hover:bg-emerald-400/10 font-bold py-3 px-8 rounded-full transition-all flex items-center justify-center space-x-2 shadow-[0_0_15px_rgba(16,185,129,0.1)] hover:shadow-[0_0_25px_rgba(16,185,129,0.3)] ${!isLoaded ? 'opacity-50 pointer-events-none' : ''}`}>
                <Upload className="w-5 h-5" />
                <span>Upload from Device</span>
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={handleFileUpload}
                  disabled={!isLoaded}
                />
              </label>
            </div>
          </div>
        )}

        {/* Live HUD Overlay */}
        {isCameraActive && (
          <div className="absolute inset-0 p-4 sm:p-6 pointer-events-none flex flex-col justify-between z-20">

            {/* Top Bar: Warnings & Settings */}
            <div className="flex justify-between items-start w-full">
              <div className="flex-1 space-y-2">
              </div>
              {!isRecording && (
                <button
                  onClick={() => setShowSettings(true)}
                  className="pointer-events-auto bg-zinc-900/80 hover:bg-zinc-800 backdrop-blur-md border border-zinc-700 text-white p-2 rounded-full transition-all shadow-lg ml-4"
                >
                  <Settings2 className="w-5 h-5" />
                </button>
              )}
            </div>

            {/* Metrics Display */}
            <div className="absolute top-16 left-4 sm:top-20 sm:left-6 flex flex-col gap-3 sm:gap-4 pointer-events-none">
              {selectedMetrics.map(id => {
                const def = AVAILABLE_METRICS.find(m => m.id === id)!;
                const val = metricsValues[id] || 0;
                const unitStr = def.type === 'angle' ? '°' : linearUnit;

                let highlight = activeHighlights[id] || false;
                if (id === 'shoulderTurn' && val >= 90) highlight = true;

                return (
                  <HUDBox
                    key={id}
                    label={def.label}
                    value={`${val.toFixed(def.type === 'angle' ? 0 : 1)}${unitStr}`}
                    highlight={highlight}
                  />
                );
              })}
            </div>

            {/* Bottom Controls (Center) */}
            <div className="absolute bottom-4 left-0 right-0 px-4 pointer-events-auto flex flex-col items-center space-y-4">
              {isVideoFile && !isRecording && (
                <div className="w-full max-w-md flex items-center space-x-3 bg-zinc-900/80 backdrop-blur-md p-2 rounded-full border border-zinc-700 shadow-lg">
                  <button
                    onClick={() => {
                      if (videoRef.current) {
                        if (isPlaying) videoRef.current.pause();
                        else videoRef.current.play();
                        setIsPlaying(!isPlaying);
                      }
                    }}
                    className="text-white p-1 hover:text-emerald-400 transition-colors"
                  >
                    {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                  </button>
                  <input
                    type="range"
                    min="0"
                    max={videoDuration || 100}
                    step="0.01"
                    value={videoProgress}
                    onChange={handleScrub}
                    className="flex-1 h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>
              )}
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setIsRecording(!isRecording)}
                  className={`backdrop-blur-md border font-medium py-2 px-6 rounded-full transition-all flex items-center space-x-2 shadow-lg ${
                    isRecording
                      ? 'bg-red-500/20 border-red-500/50 text-red-400 hover:bg-red-500/30'
                      : 'bg-zinc-900/80 border-zinc-700 text-white hover:bg-zinc-800'
                  }`}
                >
                  <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-400 animate-pulse' : 'bg-red-500'}`} />
                  <span className="text-sm sm:text-base whitespace-nowrap">{isRecording ? 'Stop Recording' : 'Record'}</span>
                </button>

                {!isRecording && (
                  <div className="flex space-x-2">
                    <button
                      onClick={calibrateAddress}
                      className="bg-zinc-900/80 hover:bg-zinc-800 backdrop-blur-md border border-zinc-700 text-white font-medium py-2 px-6 rounded-full transition-all flex items-center space-x-2 shadow-lg"
                    >
                      <RefreshCw className={`w-4 h-4 ${addressFeedback ? 'text-emerald-400' : ''}`} />
                      <span className={`text-sm sm:text-base whitespace-nowrap ${addressFeedback ? 'text-emerald-400' : ''}`}>
                        {addressFeedback ? 'Address Set!' : 'Set Address'}
                      </span>
                    </button>
                    {addressState && (
                      <button
                        onClick={clearAddress}
                        className="bg-zinc-900/80 hover:bg-zinc-800 backdrop-blur-md border border-zinc-700 text-zinc-400 hover:text-white font-medium py-2 px-4 rounded-full transition-all flex items-center shadow-lg"
                      >
                        <span className="text-sm sm:text-base whitespace-nowrap">Clear</span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Controls */}
      {isCameraActive && (
        <div className="p-4 bg-zinc-950 border-t border-zinc-800 flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <div className="text-xs text-zinc-500 font-mono">
              EMA: {EMA_ALPHA} | HUD: {1000/HUD_UPDATE_INTERVAL_MS}fps
            </div>
            <div className={`text-xs font-mono font-semibold px-2 py-1 rounded ${landmarksDetected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
              Landmarks Detected: {landmarksDetected ? 'YES' : 'NO'}
            </div>
          </div>
          <button
            onClick={stopCamera}
            className="text-red-400 hover:text-red-300 text-sm font-medium transition-colors"
          >
            {isVideoFile ? 'Stop Video' : 'Stop Camera'}
          </button>
        </div>
      )}
    </div>
  );
}

function HUDBox({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`
      backdrop-blur-md rounded-2xl p-3 sm:p-4 border min-w-[120px] sm:min-w-[160px] transition-all duration-300
      ${highlight
        ? 'bg-emerald-500/20 border-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.4)] scale-105'
        : 'bg-slate-950/80 border-slate-800/80'}
    `}>
      <div className="flex items-center justify-between mb-1 sm:mb-2">
        <div className="text-[10px] sm:text-xs uppercase tracking-wider text-slate-400 font-semibold">{label}</div>
        {highlight && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
      </div>
      <div className={`text-2xl sm:text-4xl font-mono font-bold tracking-tight ${highlight ? 'text-emerald-400' : 'text-white'}`}>
        {value}
      </div>
    </div>
  );
}
