import { useState, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Upload, ImageIcon, Loader2 } from "lucide-react";

interface XRayUploaderProps {
  onImageUpload: (file: File) => void;
  isAnalyzing: boolean;
}

export function XRayUploader({ onImageUpload, isAnalyzing }: XRayUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onImageUpload(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) onImageUpload(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  return (
    <div
      onClick={() => !isAnalyzing && inputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden
        ${isDragging
          ? "border-cyan-400 bg-cyan-400/10 shadow-[0_0_40px_rgba(34,211,238,0.2)]"
          : "border-white/20 bg-white/5 hover:border-cyan-500/50 hover:bg-cyan-500/5 hover:shadow-[0_0_30px_rgba(34,211,238,0.1)]"
        }`}
    >
      {/* Animated corner accents */}
      {["top-0 left-0", "top-0 right-0 rotate-90", "bottom-0 right-0 rotate-180", "bottom-0 left-0 -rotate-90"].map((pos, i) => (
        <span
          key={i}
          className={`absolute ${pos} w-5 h-5 border-t-2 border-l-2 ${isDragging ? "border-cyan-400" : "border-cyan-500/40"} transition-colors duration-300`}
        />
      ))}

      <div className="flex flex-col items-center gap-5 py-16 px-8">
        <motion.div
          animate={isDragging ? { scale: 1.15, rotate: -5 } : { scale: 1, rotate: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          className="relative"
        >
          <div className={`w-20 h-20 rounded-2xl flex items-center justify-center transition-all duration-300
            ${isDragging
              ? "bg-cyan-400/20 shadow-[0_0_30px_rgba(34,211,238,0.4)]"
              : "bg-gradient-to-br from-cyan-500/20 to-blue-600/20"
            }`}
          >
            <AnimatePresence mode="wait">
              {isAnalyzing ? (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <Loader2 className="size-8 text-cyan-400 animate-spin" />
                </motion.div>
              ) : isDragging ? (
                <motion.div key="drag" initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
                  <ImageIcon className="size-8 text-cyan-400" />
                </motion.div>
              ) : (
                <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <Upload className="size-8 text-cyan-400" />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          {/* Pulse ring */}
          {!isAnalyzing && (
            <span className="absolute inset-0 rounded-2xl border border-cyan-400/30 animate-ping" />
          )}
        </motion.div>

        <div className="text-center">
          <p className="text-white text-base mb-1" style={{ fontWeight: 600 }}>
            {isAnalyzing ? "Analyzing X-Ray..." : isDragging ? "Drop to Analyze" : "Upload Dental X-Ray"}
          </p>
          <p className="text-white/40 text-sm">
            {isAnalyzing ? "CNN model is processing your image" : "Drag & drop or click to select an image"}
          </p>
        </div>

        {!isAnalyzing && (
          <motion.div
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 text-white text-sm shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-shadow"
            style={{ fontWeight: 600 }}
          >
            Select Image
          </motion.div>
        )}

        <p className="text-xs text-white/25 tracking-wider uppercase">
          Supports JPG · PNG · JPEG
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
        disabled={isAnalyzing}
      />
    </div>
  );
}
