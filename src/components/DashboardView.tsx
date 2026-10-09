import React from 'react';
import {
  Stethoscope,
  Layers,
  BookOpen,
  ShieldCheck,
  Building2,
  ArrowRight,
  Cpu,
  Database,
  CheckCircle2,
  AlertTriangle,
  Flame,
} from 'lucide-react';
import { ActiveTab } from './Navbar';

interface DashboardViewProps {
  onNavigate: (tab: ActiveTab) => void;
  hospitalsSummary: any;
  modelInfo: any;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onNavigate,
  hospitalsSummary,
  modelInfo,
}) => {
  return (
    
      <div className="space-y-8 pl-4 sm:pl-6 lg:pl-10">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-950 border border-slate-700/60 p-6 sm:p-10 shadow-2xl">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-80 h-80 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 -mb-20 w-80 h-80 rounded-full bg-blue-600/10 blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-4xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold tracking-wide">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            RESEARCH DEMONSTRATION & COLLEGE BENCHMARK
          </div>

          
        <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight">
  FedX-Ray
  <span className="block mt-3 text-lg sm:text-2xl font-semibold text-cyan-400">
    Privacy-Preserving Chest X-Ray Intelligence
  </span>
        </h1>



          
          <p className="text-sm sm:text-base text-slate-300 leading-relaxed max-w-3xl">
  An AI-powered chest X-ray analysis platform using Federated Learning to
  protect patient privacy and RAG to provide evidence-based medical insights.
          </p>



          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('prediction')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-sm font-semibold shadow-lg shadow-cyan-500/25 transition-all transform hover:-translate-y-0.5 cursor-pointer"
            >
              <Stethoscope className="w-4 h-4" />
              <span>Test X-Ray Prediction</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>
            <button
              onClick={() => onNavigate('federated')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700/80 border border-slate-700 text-slate-200 text-sm font-semibold transition-all cursor-pointer"
            >
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Explore Federated Architecture</span>
            </button>
            <button
              onClick={() => onNavigate('performance')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700/80 border border-slate-700 text-slate-200 text-sm font-semibold transition-all cursor-pointer"
            >
              <Database className="w-4 h-4 text-indigo-400" />
              <span>IEEE Benchmark Metrics</span>
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics Banner */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium">Dataset Pool</span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-2xl font-bold text-white">660</div>
            <p className="text-xs text-slate-400 mt-0.5">Partitioned Radiographs (224×224)</p>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium">Hospital Nodes</span>
            <Building2 className="w-4 h-4 text-blue-400" />
          </div>
          <div>
            <div className="text-2xl font-bold text-white">3 Nodes</div>
            <p className="text-xs text-slate-400 mt-0.5">Non-IID Partitioned Silos</p>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium">Diagnostic Scope</span>
            <Stethoscope className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <div className="text-2xl font-bold text-white">6 Classes</div>
            <p className="text-xs text-slate-400 mt-0.5">Normal + 5 Acute Pathologies</p>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium">Privacy Status</span>
            <ShieldCheck className="w-4 h-4 text-teal-400" />
          </div>
          <div>
            <div className="text-2xl font-bold text-emerald-400">0 Images Sent</div>
            <p className="text-xs text-slate-400 mt-0.5">Weights Only Aggregation</p>
          </div>
        </div>
      </div>

      {/* 4 Pillars Card Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Pillar 1: Federated Learning */}
        <div
          onClick={() => onNavigate('federated')}
          className="group p-6 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-cyan-500/40 transition-all cursor-pointer shadow-lg hover:shadow-cyan-500/10 flex flex-col justify-between"
        >
          <div>
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-110 transition-transform">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white group-hover:text-cyan-400 transition-colors">
              1. Federated Learning Architecture
            </h3>
            <p className="text-sm text-slate-300 mt-2 leading-relaxed">
              Hospital A (Pulmonology), Hospital B (Cardiology), and Hospital C (General) perform
              local gradient updates on their private data. The central federated coordinator executes
              Federated Averaging (FedAvg), distributing an updated global model every round.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Non-IID Partitions
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                FedAvg Aggregator
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Multi-Round Convergence
              </span>
            </div>
          </div>
          <div className="mt-6 flex items-center gap-2 text-xs font-semibold text-cyan-400">
            <span>Explore Federated Workflow</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>

        {/* Pillar 2: Chest X-ray Prediction & Grad-CAM */}
        <div
          onClick={() => onNavigate('prediction')}
          className="group p-6 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-blue-500/40 transition-all cursor-pointer shadow-lg hover:shadow-blue-500/10 flex flex-col justify-between"
        >
          <div>
            <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-4 group-hover:scale-110 transition-transform">
              <Stethoscope className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white group-hover:text-blue-400 transition-colors">
              2. Chest X-Ray Disease Detection & Grad-CAM
            </h3>
            <p className="text-sm text-slate-300 mt-2 leading-relaxed">
              Upload any chest radiograph or choose verified clinical test cases. The trained
              Federated Global Model predicts the condition with calibrated probabilities and
              generates a Grad-CAM attention heatmap overlay.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Normal / Abnormal Status
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Ranked Differential Probabilities
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Visual Grad-CAM Attention
              </span>
            </div>
          </div>
          <div className="mt-6 flex items-center gap-2 text-xs font-semibold text-blue-400">
            <span>Analyze a Radiograph</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>

        {/* Pillar 3: Medical RAG */}
        <div
          onClick={() => onNavigate('rag')}
          className="group p-6 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-emerald-500/40 transition-all cursor-pointer shadow-lg hover:shadow-emerald-500/10 flex flex-col justify-between"
        >
          <div>
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mb-4 group-hover:scale-110 transition-transform">
              <BookOpen className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white group-hover:text-emerald-400 transition-colors">
              3. RAG-Based Medical Information Retrieval
            </h3>
            <p className="text-sm text-slate-300 mt-2 leading-relaxed">
              After diagnosis, the predicted class automatically queries a controlled knowledge
              base of peer-reviewed clinical documents (WHO, CDC, NIH, AHA, BTS). Ask interactive
              questions with answers grounded in documented sources.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                WHO / CDC / NIH Knowledge Base
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Symptoms, Causes & Treatments
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Interactive Q&A
              </span>
            </div>
          </div>
          <div className="mt-6 flex items-center gap-2 text-xs font-semibold text-emerald-400">
            <span>Browse Medical Intelligence</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>

        {/* Pillar 4: Patient Privacy */}
        <div
          onClick={() => onNavigate('privacy')}
          className="group p-6 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-purple-500/40 transition-all cursor-pointer shadow-lg hover:shadow-purple-500/10 flex flex-col justify-between"
        >
          <div>
            <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 mb-4 group-hover:scale-110 transition-transform">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white group-hover:text-purple-400 transition-colors">
              4. Patient Privacy & Threat Model
            </h3>
            <p className="text-sm text-slate-300 mt-2 leading-relaxed">
              Patient radiographs never leave hospital grounds. Hospitals compute weight deltas
              locally and transmit only anonymous numerical parameters. Includes transparent research
              disclosure on FedAvg privacy boundaries.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Zero Image Sharing
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Institutional Data Sovereignty
              </span>
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                Honest Threat Model
              </span>
            </div>
          </div>
          <div className="mt-6 flex items-center gap-2 text-xs font-semibold text-purple-400">
            <span>Read Privacy Protocol</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>
      </div>

      {/* Workflow Diagram Banner */}
      <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-4">
        <h3 className="text-base font-semibold text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          <span>Federated Learning Clinical Workflow</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-xs">
          <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700 space-y-1">
            <span className="font-bold text-cyan-400">Step 1</span>
            <p className="font-medium text-white">Global Model Init</p>
            <p className="text-slate-400 text-[11px]">Server initializes common FedChestNet architecture</p>
          </div>

          <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700 space-y-1">
            <span className="font-bold text-cyan-400">Step 2</span>
            <p className="font-medium text-white">Model Distribution</p>
            <p className="text-slate-400 text-[11px]">Weights broadcast to Hospital A, B, and C</p>
          </div>

          <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700 space-y-1">
            <span className="font-bold text-cyan-400">Step 3</span>
            <p className="font-medium text-white">Local SGD Training</p>
            <p className="text-slate-400 text-[11px]">Hospitals train locally on private silos (No image egress)</p>
          </div>

          <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700 space-y-1">
            <span className="font-bold text-cyan-400">Step 4</span>
            <p className="font-medium text-white">FedAvg Aggregation</p>
            <p className="text-slate-400 text-[11px]">Server combines updates: Σ (n_k / N) * W_k</p>
          </div>

          <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700 space-y-1">
            <span className="font-bold text-cyan-400">Step 5</span>
            <p className="font-medium text-white">Inference + RAG</p>
            <p className="text-slate-400 text-[11px]">Global model predicts and triggers RAG medical retrieval</p>
          </div>
        </div>
      </div>
    </div>
  );
};
