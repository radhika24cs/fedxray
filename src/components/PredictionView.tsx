import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Stethoscope,
  Sparkles,
  AlertCircle,
  Eye,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  BookOpen,
  RefreshCw,
  FileText,
} from 'lucide-react';
import { ActiveTab } from './Navbar';

interface SampleXray {
  id: string;
  condition: string;
  description: string;
  relativePath: string;
  expectedClass: string;
  dataUrl?: string;
  patientId?: string;
  patientAge?: number;
  patientGender?: string;
  viewPosition?: string;
  clinicalSource?: string;
}

interface RankedPrediction {
  condition: string;
  probability: number;
  isTop: boolean;
  category: string;
}

interface PredictionViewProps {
  onSelectConditionForRag: (condition: string) => void;
  onNavigate: (tab: ActiveTab) => void;
}

export const PredictionView: React.FC<PredictionViewProps> = ({
  onSelectConditionForRag,
  onNavigate,
}) => {
  const [samples, setSamples] = useState<SampleXray[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<any>(null);

  // Grad-CAM controls
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.65);
  const [viewMode, setViewMode] = useState<'composite' | 'original' | 'heatmap'>('composite');
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Fetch samples on mount
  useEffect(() => {
    fetch('/api/sample-xrays')
      .then((r) => r.json())
      .then((data) => {
        if (data.samples) {
          setSamples(data.samples);
          // Auto-select first sample for instant live demo experience
          if (data.samples.length > 0) {
            handleSelectSample(data.samples[0]);
          }
        }
      })
      .catch((err) => console.error('Failed to load sample x-rays:', err));
  }, []);

  const handleSelectSample = (sample: SampleXray) => {
    setSelectedSampleId(sample.id);
    setSelectedImage(sample.dataUrl || null);
    setPredictionResult(null);
    setError(null);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedSampleId(null);
    const reader = new FileReader();
    reader.onload = (uploadEvent) => {
      setSelectedImage(uploadEvent.target?.result as string);
      setPredictionResult(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const runAnalysis = async () => {
    if (!selectedImage && !selectedSampleId) {
      setError('Please select or upload a chest radiograph first.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sampleId: selectedSampleId,
          imageBase64: selectedImage,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      setPredictionResult(data);
      onSelectConditionForRag(data.predictedCondition);
    } catch (err: any) {
      setError(err.message || 'Error occurred during federated model inference.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Render Grad-CAM Heatmap onto canvas
  useEffect(() => {
    if (!predictionResult?.gradcam || !canvasRef.current || !selectedImage) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      canvas.width = 224;
      canvas.height = 224;

      // Draw original image
      if (viewMode === 'original' || viewMode === 'composite') {
        ctx.drawImage(img, 0, 0, 224, 224);
      } else {
        // Black background for standalone heatmap
        ctx.fillStyle = '#050b14';
        ctx.fillRect(0, 0, 224, 224);
      }

      // Draw Grad-CAM overlay
      if (viewMode === 'composite' || viewMode === 'heatmap') {
        const { gridSize, heatmap } = predictionResult.gradcam;
        const cellW = 224 / gridSize;
        const cellH = 224 / gridSize;

        const overlayAlpha = viewMode === 'heatmap' ? 0.95 : heatmapOpacity;

        for (let r = 0; r < gridSize; r++) {
          for (let c = 0; c < gridSize; c++) {
            const val = heatmap[r * gridSize + c];
            if (val > 0.05) {
              // Jet / Turbo colormap simulation
              // val: 0.0 -> Blue, 0.5 -> Green/Yellow, 1.0 -> Red
              const red = Math.min(255, Math.floor(Math.max(0, 2 * val - 0.5) * 255));
              const green = Math.min(255, Math.floor(Math.max(0, 1 - 2 * Math.abs(val - 0.5)) * 255));
              const blue = Math.min(255, Math.floor(Math.max(0, 1.5 - 2 * val) * 255));

              ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${val * overlayAlpha})`;
              ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
            }
          }
        }
      }
    };
    img.src = selectedImage;
  }, [predictionResult, viewMode, heatmapOpacity, selectedImage]);

 
    return (
  <div className="space-y-8">
    {/* Header */}
    <div>
      <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-1">
        <Stethoscope className="w-4 h-4" />
        <span>Inference & Visual Explainability</span>
      </div>

      <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
        AI-Powered Chest X-Ray Intelligence
      </h2>

      <p className="text-sm text-slate-300 mt-1 max-w-3xl">
        Transform chest radiographs into actionable insights with privacy-preserving
        Federated AI. Explore disease predictions, confidence scores, visual
        Grad-CAM heatmaps, and evidence-grounded clinical insights powered by
        Retrieval-Augmented Generation (RAG).
      </p>
    </div>

      {/* Quick Select Benchmark Samples */}
      <div className="p-4 sm:p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Select a Verified Clinical Radiograph for Demonstration</span>
          </span>
          <span className="text-xs text-slate-400">Held-out global test partition (N=120)</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
          {samples.map((s) => {
            const isSelected = selectedSampleId === s.id;
            return (
              <button
                key={s.id}
                onClick={() => handleSelectSample(s)}
                className={`p-2 rounded-xl text-left border transition-all cursor-pointer flex flex-col justify-between overflow-hidden group ${
                  isSelected
                    ? 'bg-cyan-500/15 border-cyan-400 text-white shadow-md shadow-cyan-500/20 ring-1 ring-cyan-400/50'
                    : 'bg-slate-800/70 border-slate-700/70 text-slate-300 hover:border-slate-500 hover:bg-slate-800'
                }`}
              >
                <div>
                  {/* Actual Radiograph Thumbnail */}
                  <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-black mb-2 border border-slate-700/60">
                    {s.dataUrl ? (
                      <img
                        src={s.dataUrl}
                        alt={s.condition}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300 filter contrast-110"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-slate-600 text-[10px]">
                        X-RAY
                      </div>
                    )}
                    <span className="absolute bottom-1 right-1 text-[9px] font-mono bg-black/80 px-1 rounded text-cyan-300">
                      {s.viewPosition?.split(' ')[0] || 'PA'}
                    </span>
                  </div>

                  <span
                    className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-bold uppercase mb-1 ${
                      s.condition === 'Normal'
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-amber-950 text-amber-400 border border-amber-800'
                    }`}
                  >
                    {s.condition}
                  </span>
                  
                  {s.patientAge && (
                    <div className="text-[10px] text-slate-400 font-mono mb-1">
                      {s.patientId} • {s.patientAge}{s.patientGender}
                    </div>
                  )}

                  <p className="text-[10px] text-slate-400 line-clamp-2 leading-snug">
                    {s.description}
                  </p>
                </div>

                {isSelected ? (
                  <div className="mt-2 text-[10px] font-semibold text-cyan-400 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Selected</span>
                  </div>
                ) : (
                  <div className="mt-2 text-[9px] text-slate-400 truncate">
                    {s.clinicalSource?.split('/')[0]?.trim()}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Interaction Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Viewer & Upload (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4 shadow-xl">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Thoracic Radiograph
              </span>
              <label className="text-xs text-cyan-400 hover:text-cyan-300 cursor-pointer font-medium flex items-center gap-1">
                <Upload className="w-3.5 h-3.5" />
                <span>Upload Custom Image</span>
                <input
                  type="file"
                  accept="image/png, image/jpeg, image/jpg"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>

            {/* Canvas or Image Display */}
            <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center group">
              {predictionResult?.gradcam ? (
                <canvas
                  ref={canvasRef}
                  className="w-full h-full object-contain filter contrast-105"
                />
              ) : selectedImage ? (
                <img
                  src={selectedImage}
                  alt="Selected Chest Radiograph"
                  className="w-full h-full object-contain filter contrast-105"
                />
              ) : (
                <div className="text-center p-6 space-y-2">
                  <Upload className="w-10 h-10 text-slate-600 mx-auto" />
                  <p className="text-xs text-slate-400">
                    Upload a 224×224 chest radiograph or choose a test sample above
                  </p>
                </div>
              )}

              {/* Watermark badge */}
              <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur text-[10px] text-slate-300 border border-slate-700/60 font-mono">
                PA CHEST CXR
              </div>

              {isAnalyzing && (
                <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center space-y-3 z-20">
                  <div className="w-10 h-10 border-3 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                  <p className="text-xs font-semibold text-cyan-400 tracking-wide uppercase animate-pulse">
                    Federated Global Model Analyzing...
                  </p>
                  <p className="text-[11px] text-slate-400">
                    Extracting feature maps & computing Grad-CAM
                  </p>
                </div>
              )}
            </div>

            {/* Grad-CAM Heatmap Controls if available */}
            {predictionResult?.gradcam && (
              <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                    <Eye className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Grad-CAM Attention Map</span>
                  </span>
                  <div className="flex items-center gap-1 bg-slate-900 rounded-lg p-0.5 border border-slate-700">
                    <button
                      onClick={() => setViewMode('composite')}
                      className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                        viewMode === 'composite'
                          ? 'bg-cyan-500 text-white'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Composite
                    </button>
                    <button
                      onClick={() => setViewMode('heatmap')}
                      className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                        viewMode === 'heatmap'
                          ? 'bg-cyan-500 text-white'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Heatmap
                    </button>
                    <button
                      onClick={() => setViewMode('original')}
                      className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                        viewMode === 'original'
                          ? 'bg-cyan-500 text-white'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Original
                    </button>
                  </div>
                </div>

                {viewMode === 'composite' && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Heatmap Opacity</span>
                      <span>{Math.round(heatmapOpacity * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={heatmapOpacity}
                      onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                    />
                  </div>
                )}

                <p className="text-[10px] text-slate-400 italic">
                  {predictionResult.gradcam.label}
                </p>
              </div>
            )}

            {/* Clinical Case Details */}
            {selectedSampleId && (
              <div className="p-3.5 rounded-xl bg-slate-800/80 border border-slate-700/80 text-xs space-y-1.5 font-mono">
                <div className="flex justify-between items-center text-slate-300">
                  <span className="text-slate-400">Archive Source:</span>
                  <span className="text-cyan-300 font-semibold truncate max-w-[210px]">
                    {samples.find((s) => s.id === selectedSampleId)?.clinicalSource || 'Kaggle Guangzhou / Hospital X-Ray'}
                  </span>
                </div>
                <div className="flex justify-between items-center text-slate-300">
                  <span className="text-slate-400">Patient Case:</span>
                  <span className="text-white">
                    {samples.find((s) => s.id === selectedSampleId)?.patientId || 'PAT_1435'} (Age: {samples.find((s) => s.id === selectedSampleId)?.patientAge || 46} • {samples.find((s) => s.id === selectedSampleId)?.patientGender || 'M'})
                  </span>
                </div>
                <div className="flex justify-between items-center text-slate-300">
                  <span className="text-slate-400">Projection View:</span>
                  <span className="text-emerald-400">
                    {samples.find((s) => s.id === selectedSampleId)?.viewPosition || 'PA Projection'} • 224×224 8-bit
                  </span>
                </div>
              </div>
            )}

            {/* Action Trigger Button */}
            <button
              onClick={runAnalysis}
              disabled={isAnalyzing || !selectedImage}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-sm font-bold shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all cursor-pointer flex items-center justify-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Analyzing Radiograph...</span>
                </>
              ) : (
                <>
                  <Stethoscope className="w-4 h-4" />
                  <span>Run Federated Inference</span>
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Prediction Output & RAG Overview (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {predictionResult ? (
            <div className="space-y-6">
              {/* Primary Diagnostic Finding Card */}
              <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 border border-slate-700 shadow-xl space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="space-y-0.5">
                    <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                      Predicted Primary Condition
                    </span>
                    <h3 className="text-3xl font-black text-white tracking-tight flex items-center gap-3">
                      <span>Possible {predictionResult.predictedCondition}</span>
                      <span
                        className={`text-xs font-bold px-2.5 py-1 rounded-full uppercase ${
                          predictionResult.predictedCondition === 'Normal'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                        }`}
                      >
                        {predictionResult.status}
                      </span>
                    </h3>
                  </div>

                  <div className="text-right">
                    <span className="text-xs text-slate-400">Confidence Score</span>
                    <div className="text-3xl font-extrabold text-cyan-400">
                      {predictionResult.confidence}%
                    </div>
                  </div>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-cyan-400 to-blue-500 h-2.5 rounded-full transition-all duration-500"
                    style={{ width: `${predictionResult.confidence}%` }}
                  />
                </div>

                {/* Ranked Possibilities Breakdown */}
                <div className="pt-2 space-y-2">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Full Differential Distribution (All 6 Classes)
                  </span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {predictionResult.rankedPredictions.map((pred: RankedPrediction) => (
                      <div
                        key={pred.condition}
                        className={`p-2.5 rounded-xl border flex items-center justify-between ${
                          pred.isTop
                            ? 'bg-cyan-500/10 border-cyan-500/40 text-white font-semibold'
                            : 'bg-slate-800/40 border-slate-700/50 text-slate-300'
                        }`}
                      >
                        <span className="text-xs flex items-center gap-1.5">
                          {pred.isTop && <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />}
                          <span>{pred.condition}</span>
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono">{pred.probability}%</span>
                          <span
                            className={`text-[9px] px-1 rounded ${
                              pred.category === 'Normal'
                                ? 'bg-emerald-950 text-emerald-400'
                                : 'bg-slate-700 text-slate-300'
                            }`}
                          >
                            {pred.category}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{predictionResult.disclaimer}</span>
                </div>
              </div>

              {/* Automated RAG Preview Box */}
              {predictionResult.ragReport && (
                <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-4 shadow-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <BookOpen className="w-5 h-5 text-emerald-400" />
                      <h4 className="text-base font-bold text-white">
                        RAG Medical Intelligence Overview
                      </h4>
                    </div>
                    <span className="text-xs px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-300 font-mono">
                      ICD-10: {predictionResult.ragReport.icd10}
                    </span>
                  </div>

                  <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                    {predictionResult.ragReport.overview}
                  </p>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                    <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-1.5">
                      <span className="text-xs font-semibold text-slate-200">Common Symptoms</span>
                      <ul className="text-xs text-slate-300 space-y-1 list-disc list-inside">
                        {predictionResult.ragReport.symptoms.slice(0, 3).map((sym: string, i: number) => (
                          <li key={i} className="line-clamp-1">{sym}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-1.5">
                      <span className="text-xs font-semibold text-slate-200">General Treatment</span>
                      <ul className="text-xs text-slate-300 space-y-1 list-disc list-inside">
                        {predictionResult.ragReport.treatmentApproaches.slice(0, 3).map((tx: string, i: number) => (
                          <li key={i} className="line-clamp-1">{tx}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  <div className="pt-2 flex items-center justify-between border-t border-slate-800">
                    <span className="text-xs text-slate-400">
                      Sources:{' '}
                      {predictionResult.ragReport.sources
                        .map((s: any) => s.organization)
                        .slice(0, 2)
                        .join(', ')}
                    </span>
                    <button
                      onClick={() => onNavigate('rag')}
                      className="inline-flex items-center gap-1.5 text-xs font-bold text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer"
                    >
                      <span>Explore Full 6-Section RAG Guide</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center p-8 rounded-2xl bg-slate-900/40 border border-slate-800/80 text-center space-y-3">
              <div className="w-16 h-16 rounded-2xl bg-slate-800/60 flex items-center justify-center text-slate-600">
                <Stethoscope className="w-8 h-8" />
              </div>
              <h3 className="text-base font-semibold text-slate-300">
                Awaiting Chest X-ray Image Analysis
              </h3>
              <p className="text-xs text-slate-400 max-w-sm">
                Select a radiograph from the test benchmark gallery on top, or upload your own image
                and click &quot;Run Federated Inference&quot;.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
