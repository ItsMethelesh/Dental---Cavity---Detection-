import { CavityDetection } from "../components/x-ray-analysis-view";

export async function analyzeDentalXRay(imageUrl: string): Promise<CavityDetection[]> {
  // Simulate CNN model processing time
  await new Promise((resolve) => setTimeout(resolve, 2000 + Math.random() * 1000));

  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;

    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      if (!ctx) {
        resolve([]);
        return;
      }

      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Get image data for analysis
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Simulate CNN-based cavity detection
      const detections: CavityDetection[] = [];
      
      // Analyze image in grid segments to find dark regions (potential cavities)
      const segmentSize = 60;
      const threshold = 80; // Darkness threshold
      const minDarkPixels = 200; // Minimum dark pixels to consider as potential cavity

      for (let y = 0; y < canvas.height - segmentSize; y += Math.floor(segmentSize / 2)) {
        for (let x = 0; x < canvas.width - segmentSize; x += Math.floor(segmentSize / 2)) {
          let darkPixelCount = 0;
          let totalIntensity = 0;
          let pixelCount = 0;

          // Analyze segment
          for (let sy = 0; sy < segmentSize && y + sy < canvas.height; sy++) {
            for (let sx = 0; sx < segmentSize && x + sx < canvas.width; sx++) {
              const i = ((y + sy) * canvas.width + (x + sx)) * 4;
              const r = data[i];
              const g = data[i + 1];
              const b = data[i + 2];
              
              // Calculate grayscale intensity
              const intensity = (r + g + b) / 3;
              totalIntensity += intensity;
              pixelCount++;

              if (intensity < threshold) {
                darkPixelCount++;
              }
            }
          }

          const avgIntensity = totalIntensity / pixelCount;
          
          // Check if this segment has enough dark pixels and appropriate characteristics
          if (darkPixelCount > minDarkPixels && avgIntensity < 120) {
            // Calculate confidence based on darkness and uniformity
            const darkRatio = darkPixelCount / pixelCount;
            const confidence = Math.min(0.95, Math.max(0.65, darkRatio * 1.2 + (1 - avgIntensity / 255) * 0.3));

            // Determine severity based on darkness and confidence
            let severity: "low" | "medium" | "high";
            if (avgIntensity < 60 && confidence > 0.85) {
              severity = "high";
            } else if (avgIntensity < 90 && confidence > 0.75) {
              severity = "medium";
            } else {
              severity = "low";
            }

            // Check for overlap with existing detections
            const hasOverlap = detections.some((existing) => {
              const overlapX = Math.max(0, Math.min(existing.x + existing.width, x + segmentSize) - Math.max(existing.x, x));
              const overlapY = Math.max(0, Math.min(existing.y + existing.height, y + segmentSize) - Math.max(existing.y, y));
              const overlapArea = overlapX * overlapY;
              const overlapRatio = overlapArea / (segmentSize * segmentSize);
              return overlapRatio > 0.3;
            });

            if (!hasOverlap && confidence > 0.68) {
              detections.push({
                x,
                y,
                width: segmentSize,
                height: segmentSize,
                confidence,
                severity,
              });
            }
          }
        }
      }

      // Sort by confidence and limit to top detections
      const sortedDetections = detections
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 8); // Limit to 8 detections

      resolve(sortedDetections);
    };

    img.onerror = () => {
      resolve([]);
    };
  });
}
