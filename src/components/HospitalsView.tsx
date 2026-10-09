import React, { useState, useEffect } from 'react';
import {
  Building2,
  Users,
  PieChart,
  ShieldCheck,
  AlertCircle,
  Database,
  BarChart2,
  CheckCircle2,
} from 'lucide-react';

export const HospitalsView: React.FC = () => {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetch('/api/hospitals')
      .then((r) => r.json())
      .then((d) => setData(d))
      .catch((err) => console.error('Failed to load hospitals data:', err));
  }, []);

  const classes = [
    'Normal',
    'Pneumonia',
    'Atelectasis',
    'Cardiomegaly',
    'Pleural Effusion',
    'Infiltration',
  ];

  return (
    
      <div className="space-y-8 pl-4 sm:pl-6 lg:pl-10">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-1">
          <Building2 className="w-4 h-4" />
          <span>Institutional Silos</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Simulated Participating Hospitals
        </h2>
        <p className="text-sm text-slate-300 mt-1 max-w-3xl">
          Three decentralized hospital institutions participating in collaborative Federated
          Learning. Each hospital maintains its own local dataset of chest radiographs with distinct
          non-IID clinical distributions.
        </p>
      </div>

      {/* Prominent Simulation Disclaimer Banner */}
      <div className="p-4 rounded-xl bg-cyan-950/40 border border-cyan-800/60 text-cyan-200 text-xs flex items-start gap-3 shadow-md">
        <ShieldCheck className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-bold text-white">
            Simulated Hospital Institutions with Partitioned Non-IID Datasets
          </p>
          <p className="text-slate-300 leading-relaxed">
            These nodes represent simulated clinical environments using partitioned authentic Kaggle
            Guangzhou (Chest X-Ray Images) and clinical thoracic radiograph archives to model
            real-world institutional specializations. No patient identifying data or raw radiographs
            are shared across institutional silos during the federated learning workflow.
          </p>
        </div>
      </div>

      {/* Hospital Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {data?.hospitals?.map((h: any) => (
          <div
            key={h.id}
            className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between space-y-6 shadow-xl hover:border-slate-700 transition-all"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                  <Building2 className="w-5 h-5" />
                </div>
                <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  {h.samples} Radiographs
                </span>
              </div>

              <div>
                <h3 className="text-base font-bold text-white">{h.name}</h3>
                <p className="text-xs text-cyan-400 font-medium mt-0.5">{h.specialization}</p>
              </div>

              <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60 text-xs space-y-2">
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">Data Privacy:</span>
                  <span className="text-emerald-400 font-medium">{h.privacyStatus}</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span className="text-slate-400">Clinical Non-IID Bias:</span>
                  <span className="text-cyan-300 font-medium">{h.nonIIDBias}</span>
                </div>
              </div>

              {/* Class Breakdown Bars */}
              <div className="space-y-2 pt-2">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Internal Class Breakdown
                </span>
                <div className="space-y-1.5">
                  {classes.map((c) => {
                    const count = h.classDistribution[c] || 0;
                    const pct = Math.round((count / h.samples) * 100);
                    return (
                      <div key={c} className="space-y-0.5">
                        <div className="flex justify-between text-[11px] text-slate-400">
                          <span>{c}</span>
                          <span className="font-mono">
                            {count} ({pct}%)
                          </span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-1.5 rounded-full ${
                              pct > 25
                                ? 'bg-cyan-400'
                                : pct > 15
                                ? 'bg-blue-500'
                                : 'bg-slate-600'
                            }`}
                            style={{ width: `${Math.min(100, pct * 2.5)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Node Status:</span>
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Connected & Active</span>
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
