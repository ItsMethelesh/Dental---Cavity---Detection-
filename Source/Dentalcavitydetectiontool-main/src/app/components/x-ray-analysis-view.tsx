import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { AlertCircle, ScanLine, Maximize2 } from "lucide-react";

export interface CavityDetection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  severity: "low" | "medium" | "high";
}

interface XRayAnalysisViewProps {
  imageUrl: string;
  detections: CavityDetection[];
  isAnalyzing: boolean;
}

export function XRayAnalysisView({ imageUrl, detections, isAnalyzing }: XRayAnalysisViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [hoveredDetection, setHoveredDetection] = useState<number | null>(null);
  const [scanLineY, setScanLineY] = useState(0);
  const animFrameRef = useRef<number>();

  // Scan line animation
  useEffect(() => {
    if (!isAnalyzing) {
      setScanLineY(0);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      return;
    }
    let startTime: number | null = null;
    const duration = 2000;
    const animate = (ts: number) => {
      if (!startTime) startTime = ts;
      const elapsed = (ts - startTime) % duration;
      setScanLineY((elapsed / duration) * 100);
      animFrameRef.current = requestAnimationFrame(animate);
    };
    animFrameRef.current = requestAnimationFrame(animate);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [isAnalyzing]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const drawAnalysis = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx.drawImage(image, 0, 0);

      detections.forEach((detection, index) => {
        const isHovered = hoveredDetection === index;

        const overlayColors = {
          low: isHovered ? "rgba(250, 204, 21, 0.35)" : "rgba(250, 204, 21, 0.2)",
          medium: isHovered ? "rgba(251, 146, 60, 0.4)" : "rgba(251, 146, 60, 0.22)",
          high: isHovered ? "rgba(239, 68, 68, 0.45)" : "rgba(239, 68, 68, 0.25)",
        };
        const glowColors = {
          low: "#FCD34D",
          medium: "#FB923C",
          high: "#EF4444",
        };

        // Glow shadow
        ctx.shadowColor = glowColors[detection.severity];
        ctx.shadowBlur = isHovered ? 20 : 10;

        // Fill rect
        ctx.fillStyle = overlayColors[detection.severity];
        ctx.fillRect(detection.x, detection.y, detection.width, detection.height);

        // Stroke
        ctx.strokeStyle = glowColors[detection.severity];
        ctx.lineWidth = isHovered ? 3 : 1.5;
        ctx.strokeRect(detection.x, detection.y, detection.width, detection.height);

        ctx.shadowBlur = 0;

        // Corner brackets
        const bSize = Math.min(detection.width, detection.height) * 0.25;
        ctx.strokeStyle = glowColors[detection.severity];
        ctx.lineWidth = isHovered ? 3 : 2;
        const corners = [
          [detection.x, detection.y, 1, 1],
          [detection.x + detection.width, detection.y, -1, 1],
          [detection.x + detection.width, detection.y + detection.height, -1, -1],
          [detection.x, detection.y + detection.height, 1, -1],
        ] as const;
        corners.forEach(([cx, cy, dx, dy]) => {
          ctx.beginPath();
          ctx.moveTo(cx + dx * bSize, cy);
          ctx.lineTo(cx, cy);
          ctx.lineTo(cx, cy + dy * bSize);
          ctx.stroke();
        });

        // Label pill
        const label = `${(detection.confidence * 100).toFixed(0)}%`;
        const pillW = 52;
        const pillH = 20;
        const pillX = detection.x + 4;
        const pillY = detection.y - pillH - 4;

        ctx.fillStyle = glowColors[detection.severity];
        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, 5);
        ctx.fill();

        ctx.fillStyle = "#000";
        ctx.font = `bold 11px monospace`;
        ctx.textAlign = "left";
        ctx.fillText(label, pillX + 6, pillY + 13);

        const sevLabel = detection.severity.toUpperCase();
        ctx.font = `bold 9px monospace`;
        ctx.fillText(sevLabel, pillX + 28, pillY + 13);
      });
    };

    if (image.complete) drawAnalysis();
  }, [detections, hoveredDetection, imageUrl]);

  const handleImageLoad = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    ctx.drawImage(image, 0, 0);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const idx = detections.findIndex(
      (d) => x >= d.x && x <= d.x + d.width && y >= d.y && y <= d.y + d.height
    );
    setHoveredDetection(idx >= 0 ? idx : null);
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm overflow-hidden">
      {/* Card Header */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/10">
        <div className="flex items-center gap-2">
          <ScanLine className="size-4 text-cyan-400" />
          <span className="text-sm text-white/70" style={{ fontWeight: 500 }}>
            X-Ray Analysis View
          </span>
        </div>
        <div className="flex items-center gap-3">
          {detections.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-amber-400">
              <AlertCircle className="size-3.5" />
              <span>{detections.length} region{detections.length !== 1 ? "s" : ""} detected</span>
            </div>
          )}
          <Maximize2 className="size-3.5 text-white/30" />
        </div>
      </div>

      {/* Canvas Area */}
      <div className="relative p-4">
        {/* Scanning overlay */}
        <AnimatePresence>
          {isAnalyzing && (
            <motion.div
              key="scanning"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-4 z-20 rounded-lg overflow-hidden pointer-events-none"
            >
              {/* Dim overlay */}
              <div className="absolute inset-0 bg-[#020b18]/60" />
              {/* Scan line */}
              <div
                className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_12px_rgba(34,211,238,0.9)]"
                style={{ top: `${scanLineY}%`, transition: "top 0.05s linear" }}
              />
              {/* Scan label */}
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                <div className="flex gap-2 items-center px-4 py-2 rounded-xl border border-cyan-500/30 bg-black/50 backdrop-blur">
                  <div className="size-2 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="text-cyan-300 text-sm tracking-widest uppercase" style={{ fontWeight: 600 }}>
                    Scanning...
                  </span>
                </div>
                <div className="flex gap-3">
                  {["Preprocessing", "CNN Inference", "Mapping"].map((step, i) => (
                    <motion.div
                      key={step}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.2 }}
                      className="flex items-center gap-1.5 text-[11px] text-white/50"
                    >
                      <motion.span
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.2, delay: i * 0.3, repeat: Infinity }}
                        className="size-1.5 rounded-full bg-cyan-400"
                      />
                      {step}
                    </motion.div>
                  ))}
                </div>
              </div>
              {/* Corner brackets */}
              {["top-3 left-3", "top-3 right-3 rotate-90", "bottom-3 right-3 rotate-180", "bottom-3 left-3 -rotate-90"].map((p, i) => (
                <span key={i} className={`absolute ${p} w-6 h-6 border-t-2 border-l-2 border-cyan-400 opacity-80`} />
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        <img
          ref={imageRef}
          src={imageUrl}
          alt="Dental X-Ray"
          className="hidden"
          onLoad={handleImageLoad}
        />

        <canvas
          ref={canvasRef}
          className={`w-full h-auto rounded-lg transition-all duration-300 ${hoveredDetection !== null ? "cursor-crosshair" : "cursor-default"}`}
          onMouseMove={handleCanvasMouseMove}
          onMouseLeave={() => setHoveredDetection(null)}
          style={{
            filter: "brightness(1.05) contrast(1.1)",
            boxShadow: detections.length > 0
              ? "0 0 30px rgba(34, 211, 238, 0.08), inset 0 0 0 1px rgba(255,255,255,0.05)"
              : "none",
          }}
        />
      </div>

      {/* Legend */}
      {detections.length > 0 && !isAnalyzing && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-5 px-5 py-3 border-t border-white/10 bg-white/3"
        >
          <span className="text-xs text-white/40 uppercase tracking-wider">Severity</span>
          {[
            { label: "High", color: "bg-red-500", glow: "shadow-red-500/50" },
            { label: "Medium", color: "bg-orange-400", glow: "shadow-orange-400/50" },
            { label: "Low", color: "bg-yellow-400", glow: "shadow-yellow-400/50" },
          ].map((s) => (
            <div key={s.label} className="flex items-center gap-1.5">
              <span className={`size-2.5 rounded-sm ${s.color} shadow-sm ${s.glow}`} />
              <span className="text-xs text-white/50">{s.label}</span>
            </div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
