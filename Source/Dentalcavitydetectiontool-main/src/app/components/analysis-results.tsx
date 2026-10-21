import { motion } from "motion/react";
import { CavityDetection } from "./x-ray-analysis-view";
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  TrendingUp,
  Info,
} from "lucide-react";

interface AnalysisResultsProps {
  detections: CavityDetection[];
  processingTime: number;
}

export function AnalysisResults({ detections, processingTime }: AnalysisResultsProps) {
  const highSeverity = detections.filter((d) => d.severity === "high").length;
  const mediumSeverity = detections.filter((d) => d.severity === "medium").length;
  const lowSeverity = detections.filter((d) => d.severity === "low").length;

  const overallConfidence =
    detections.length > 0
      ? detections.reduce((acc, d) => acc + d.confidence, 0) / detections.length
      : 0;

  const getSeverityConfig = (severity: string) => {
    switch (severity) {
      case "high":
        return {
          icon: <XCircle className="size-4 text-red-400" />,
          label: "HIGH",
          bg: "bg-red-500/10",
          border: "border-red-500/25",
          text: "text-red-400",
          bar: "bg-gradient-to-r from-red-500 to-rose-500",
          glow: "shadow-red-500/30",
        };
      case "medium":
        return {
          icon: <AlertTriangle className="size-4 text-orange-400" />,
          label: "MED",
          bg: "bg-orange-500/10",
          border: "border-orange-500/25",
          text: "text-orange-400",
          bar: "bg-gradient-to-r from-orange-500 to-amber-500",
          glow: "shadow-orange-500/30",
        };
      case "low":
        return {
          icon: <CheckCircle className="size-4 text-yellow-400" />,
          label: "LOW",
          bg: "bg-yellow-500/10",
          border: "border-yellow-500/25",
          text: "text-yellow-400",
          bar: "bg-gradient-to-r from-yellow-400 to-amber-400",
          glow: "shadow-yellow-400/30",
        };
      default:
        return {
          icon: null,
          label: "",
          bg: "bg-white/5",
          border: "border-white/10",
          text: "text-white/50",
          bar: "bg-white/20",
          glow: "",
        };
    }
  };

  const confidenceColor =
    overallConfidence > 0.75
      ? "from-red-500 to-rose-500"
      : overallConfidence > 0.5
      ? "from-orange-500 to-amber-500"
      : "from-emerald-500 to-teal-400";

  return (
    <div className="space-y-4">
      {/* Summary Card */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm overflow-hidden"
      >
        <div className="flex items-center gap-2 px-5 py-3.5 border-b border-white/10">
          <TrendingUp className="size-4 text-cyan-400" />
          <span className="text-sm text-white/70" style={{ fontWeight: 500 }}>Analysis Summary</span>
        </div>

        <div className="p-5 space-y-5">
          {/* Confidence Bar */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-white/50 uppercase tracking-wider">Overall Confidence</span>
              <span className="text-sm text-white" style={{ fontWeight: 700 }}>
                {(overallConfidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
              <motion.div
                className={`h-full rounded-full bg-gradient-to-r ${confidenceColor}`}
                initial={{ width: 0 }}
                animate={{ width: `${overallConfidence * 100}%` }}
                transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
                style={{
                  boxShadow: overallConfidence > 0 ? "0 0 8px rgba(34,211,238,0.4)" : "none",
                }}
              />
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-3 gap-2">
            {[
              { count: highSeverity, label: "High", bg: "bg-red-500/15", border: "border-red-500/25", text: "text-red-400" },
              { count: mediumSeverity, label: "Medium", bg: "bg-orange-500/15", border: "border-orange-500/25", text: "text-orange-400" },
              { count: lowSeverity, label: "Low", bg: "bg-yellow-500/15", border: "border-yellow-500/25", text: "text-yellow-400" },
            ].map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 + i * 0.08 }}
                className={`text-center p-3 rounded-xl border ${stat.bg} ${stat.border}`}
              >
                <div className={`text-2xl ${stat.text}`} style={{ fontWeight: 800 }}>
                  {stat.count}
                </div>
                <div className="text-[10px] text-white/40 mt-0.5 uppercase tracking-wider">{stat.label}</div>
              </motion.div>
            ))}
          </div>

          {/* Processing time */}
          <div className="flex items-center gap-2 text-xs text-white/35">
            <Clock className="size-3.5" />
            <span>Processed in {processingTime.toFixed(2)}s</span>
          </div>
        </div>
      </motion.div>

      {/* Detected Regions */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm overflow-hidden"
      >
        <div className="flex items-center gap-2 px-5 py-3.5 border-b border-white/10">
          <AlertTriangle className="size-4 text-cyan-400" />
          <span className="text-sm text-white/70" style={{ fontWeight: 500 }}>Detected Regions</span>
          {detections.length > 0 && (
            <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-cyan-500/15 border border-cyan-500/25 text-cyan-400">
              {detections.length}
            </span>
          )}
        </div>

        <div className="p-4">
          {detections.length === 0 ? (
            <div className="text-center py-10">
              <div className="size-14 rounded-full bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center mx-auto mb-3">
                <CheckCircle className="size-7 text-emerald-400" />
              </div>
              <p className="text-white/70 text-sm" style={{ fontWeight: 500 }}>No Cavities Detected</p>
              <p className="text-white/35 text-xs mt-1">The x-ray appears to be clear</p>
            </div>
          ) : (
            <div className="space-y-2.5">
              {detections.map((detection, index) => {
                const cfg = getSeverityConfig(detection.severity);
                const confPct = detection.confidence * 100;
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.15 + index * 0.06 }}
                    whileHover={{ scale: 1.01, x: 2 }}
                    className={`p-3.5 rounded-xl border ${cfg.bg} ${cfg.border} transition-all`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {cfg.icon}
                        <div>
                          <span className="text-white/80 text-sm" style={{ fontWeight: 600 }}>
                            Region {index + 1}
                          </span>
                          <p className="text-white/30 text-[11px] font-mono mt-0.5">
                            ({detection.x.toFixed(0)}, {detection.y.toFixed(0)}) · {detection.width.toFixed(0)}×{detection.height.toFixed(0)}px
                          </p>
                        </div>
                      </div>
                      <span className={`text-[10px] px-2 py-0.5 rounded-full border ${cfg.bg} ${cfg.border} ${cfg.text} uppercase tracking-wider`} style={{ fontWeight: 700 }}>
                        {cfg.label}
                      </span>
                    </div>

                    {/* Confidence mini-bar */}
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                        <motion.div
                          className={`h-full rounded-full ${cfg.bar} shadow-sm ${cfg.glow}`}
                          initial={{ width: 0 }}
                          animate={{ width: `${confPct}%` }}
                          transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 + index * 0.05 }}
                        />
                      </div>
                      <span className={`text-xs ${cfg.text}`} style={{ fontWeight: 700 }}>
                        {confPct.toFixed(0)}%
                      </span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>
      </motion.div>

      {/* Disclaimer */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="rounded-2xl border border-blue-500/20 bg-blue-500/8 p-4 flex gap-3"
      >
        <Info className="size-4 text-blue-400 shrink-0 mt-0.5" />
        <div>
          <p className="text-xs text-blue-300 mb-1" style={{ fontWeight: 600 }}>Medical Disclaimer</p>
          <p className="text-[11px] text-blue-300/60 leading-relaxed">
            This AI analysis is for educational and screening purposes only. Always consult a qualified dentist for professional diagnosis.
          </p>
        </div>
      </motion.div>
    </div>
  );
}
