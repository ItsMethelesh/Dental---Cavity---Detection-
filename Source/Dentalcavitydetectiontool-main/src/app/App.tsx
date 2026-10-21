import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Activity, RotateCcw, Cpu, Shield, Zap } from "lucide-react";
import { XRayUploader } from "./components/x-ray-uploader";
import { XRayAnalysisView, CavityDetection } from "./components/x-ray-analysis-view";
import { AnalysisResults } from "./components/analysis-results";
import { analyzeDentalXRay } from "./utils/cavity-detection";
import { toast } from "sonner";
import { Toaster } from "./components/ui/sonner";

const BG_IMAGE = "https://images.unsplash.com/photo-1629487016281-ff8f1fc9c33d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkZW50YWwlMjB4cmF5JTIwdGVldGglMjBtZWRpY2FsJTIwYmx1ZSUyMGRhcmslMjBiYWNrZ3JvdW5kfGVufDF8fHx8MTc3MjAzNjgyOXww&ixlib=rb-4.1.0&q=80&w=1080";

export default function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [detections, setDetections] = useState<CavityDetection[]>([]);
  const [processingTime, setProcessingTime] = useState(0);

  const handleImageUpload = async (file: File) => {
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setDetections([]);
    setIsAnalyzing(true);
    const startTime = performance.now();

    try {
      const results = await analyzeDentalXRay(url);
      const endTime = performance.now();
      setDetections(results);
      setProcessingTime((endTime - startTime) / 1000);

      if (results.length === 0) {
        toast.success("Analysis complete — No cavities detected");
      } else {
        toast.warning(`Analysis complete — ${results.length} potential cavity region${results.length !== 1 ? "s" : ""} detected`);
      }
    } catch (error) {
      toast.error("Error analyzing image. Please try again.");
      console.error("Analysis error:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(null);
    setDetections([]);
    setProcessingTime(0);
    setIsAnalyzing(false);
  };

  return (
    <div className="relative min-h-screen overflow-hidden text-white">
      {/* Background */}
      <div
        className="fixed inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url('${BG_IMAGE}')` }}
      />
      <div className="fixed inset-0 bg-gradient-to-br from-[#020b18]/92 via-[#041428]/88 to-[#060d1f]/95" />
      {/* Glow orbs */}
      <div className="fixed top-[-10%] left-[-5%] w-[500px] h-[500px] rounded-full bg-cyan-500/10 blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-10%] right-[-5%] w-[600px] h-[600px] rounded-full bg-blue-600/10 blur-[150px] pointer-events-none" />
      <div className="fixed top-[40%] right-[10%] w-[300px] h-[300px] rounded-full bg-teal-400/6 blur-[100px] pointer-events-none" />

      <Toaster theme="dark" />

      {/* Header */}
      <header className="relative z-20 border-b border-white/10 bg-white/5 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <motion.div
              className="flex items-center gap-3"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="relative">
                <div className="bg-gradient-to-br from-cyan-400 to-blue-500 p-2.5 rounded-xl shadow-lg shadow-cyan-500/25">
                  <Activity className="size-5 text-white" />
                </div>
                <span className="absolute -top-1 -right-1 size-2.5 bg-cyan-400 rounded-full animate-pulse" />
              </div>
              <div>
                <h1 className="text-lg text-white" style={{ fontWeight: 700, letterSpacing: "-0.02em" }}>
                  Dental<span className="text-cyan-400">AI</span>
                </h1>
                <p className="text-[11px] text-cyan-300/70 tracking-widest uppercase">
                  Cavity Detection System
                </p>
              </div>
            </motion.div>

            <motion.div
              className="flex items-center gap-4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* Live badge */}
              <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full border border-cyan-500/30 bg-cyan-500/10">
                <span className="size-1.5 rounded-full bg-cyan-400 animate-pulse" />
                <span className="text-[11px] text-cyan-300 uppercase tracking-wider">AI Engine Live</span>
              </div>

              <AnimatePresence>
                {imageUrl && (
                  <motion.button
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.85 }}
                    onClick={handleReset}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg border border-white/20 bg-white/10 hover:bg-white/15 transition-all text-sm text-white/80 hover:text-white backdrop-blur-sm"
                  >
                    <RotateCcw className="size-3.5" />
                    New Scan
                  </motion.button>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <AnimatePresence mode="wait">
          {!imageUrl ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="max-w-2xl mx-auto"
            >
              {/* Hero Text */}
              <div className="text-center mb-10">
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 mb-5"
                >
                  <Cpu className="size-3.5 text-cyan-400" />
                  <span className="text-xs text-cyan-300 tracking-wider uppercase">CNN-Powered Analysis</span>
                </motion.div>
                <motion.h2
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.15 }}
                  className="text-white mb-3"
                  style={{ fontSize: "2rem", fontWeight: 800, lineHeight: 1.2 }}
                >
                  Upload a{" "}
                  <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                    Dental X-Ray
                  </span>
                </motion.h2>
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                  className="text-white/50 text-sm"
                >
                  Our AI model scans your x-ray using deep learning to detect potential cavity regions with confidence scoring
                </motion.p>
              </div>

              <XRayUploader onImageUpload={handleImageUpload} isAnalyzing={isAnalyzing} />

              {/* Feature Cards */}
              <motion.div
                className="mt-10 grid grid-cols-1 sm:grid-cols-3 gap-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                {[
                  {
                    icon: <Cpu className="size-5 text-cyan-400" />,
                    color: "from-cyan-500/20 to-blue-500/10",
                    border: "border-cyan-500/20",
                    title: "CNN Detection",
                    desc: "Deep learning model trained on thousands of dental x-rays",
                  },
                  {
                    icon: <Zap className="size-5 text-yellow-400" />,
                    color: "from-yellow-500/15 to-orange-500/10",
                    border: "border-yellow-500/20",
                    title: "Instant Results",
                    desc: "Get precise region highlighting in seconds",
                  },
                  {
                    icon: <Shield className="size-5 text-emerald-400" />,
                    color: "from-emerald-500/15 to-teal-500/10",
                    border: "border-emerald-500/20",
                    title: "Severity Scoring",
                    desc: "Low, medium, high classification for prioritization",
                  },
                ].map((f, i) => (
                  <motion.div
                    key={f.title}
                    whileHover={{ translateY: -3, scale: 1.01 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    className={`p-5 rounded-2xl border bg-gradient-to-br backdrop-blur-sm ${f.border} ${f.color}`}
                  >
                    <div className="mb-3">{f.icon}</div>
                    <h3 className="text-white text-sm mb-1" style={{ fontWeight: 600 }}>{f.title}</h3>
                    <p className="text-white/45 text-xs leading-relaxed">{f.desc}</p>
                  </motion.div>
                ))}
              </motion.div>
            </motion.div>
          ) : (
            <motion.div
              key="analysis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
              <div className="lg:col-span-2">
                <XRayAnalysisView
                  imageUrl={imageUrl}
                  detections={detections}
                  isAnalyzing={isAnalyzing}
                />
              </div>
              <div className="lg:col-span-1">
                <AnimatePresence>
                  {!isAnalyzing && (
                    <motion.div
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ duration: 0.35 }}
                    >
                      <AnalysisResults
                        detections={detections}
                        processingTime={processingTime}
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 mt-8">
        <div className="border-t border-white/10 pt-6 text-center">
          <p className="text-xs text-white/30">
            DentalAI uses computer vision and deep learning for cavity detection.
          </p>
          <p className="text-xs text-white/20 mt-1">
            For educational purposes only — does not replace professional dental examination.
          </p>
        </div>
      </footer>
    </div>
  );
}
