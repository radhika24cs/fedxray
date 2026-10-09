
import React from 'react';
import {
  ShieldCheck,
  Lock,
  Server,
  CheckCircle2,
  FileCode2,
  Database,
  ArrowRight,
} from 'lucide-react';

export const PrivacyView: React.FC = () => {
  return (
    <div className="space-y-8 pl-4 sm:pl-6 lg:pl-10">
      {/* Header */}
      <div>
        <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-cyan-400">
          <ShieldCheck className="h-4 w-4" />
          <span>Patient Data Sovereignty</span>
        </div>

        <h2 className="text-2xl font-extrabold tracking-tight text-white sm:text-3xl">
          How Patient Privacy is Protected in Federated Learning
        </h2>

        <p className="mt-1 max-w-3xl text-sm text-slate-300">
          A transparent examination of the privacy guarantees, data flow
          boundaries, and research limitations of collaborative Federated
          Averaging across distributed clinical hospital nodes.
        </p>
      </div>

      {/* Three Core Privacy Tenets */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        {/* Card 1 */}
        <div className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/90 p-6 shadow-xl">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-400">
            <Database className="h-6 w-6" />
          </div>

          <h3 className="text-base font-bold text-white">
            1. Localized Institutional Storage
          </h3>

          <p className="text-xs leading-relaxed text-slate-300">
            Patient chest radiographs and clinical records remain within each
            hospital's local environment. Raw DICOM or PNG images are not
            intended to be transferred to the central aggregation server.
          </p>
        </div>

        {/* Card 2 */}
        <div className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/90 p-6 shadow-xl">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-blue-500/20 bg-blue-500/10 text-blue-400">
            <Lock className="h-6 w-6" />
          </div>

          <h3 className="text-base font-bold text-white">
            2. Parameter-Only Aggregation
          </h3>

          <p className="text-xs leading-relaxed text-slate-300">
            Federated clients share numerical model updates with the aggregation
            server rather than directly sharing raw patient images. Model
            updates can still carry privacy risks if additional protections
            are not applied.
          </p>
        </div>

        {/* Card 3 */}
        <div className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/90 p-6 shadow-xl">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-emerald-500/20 bg-emerald-500/10 text-emerald-400">
            <Server className="h-6 w-6" />
          </div>

          <h3 className="text-base font-bold text-white">
            3. Weighted FedAvg Consensus
          </h3>

          <p className="text-xs leading-relaxed text-slate-300">
            The federated server combines participating hospitals' model
            parameters using sample-weighted Federated Averaging (FedAvg).
            This enables collaborative model training without pooling the
            hospitals' raw source images on the central server.
          </p>
        </div>
      </div>

      {/* Privacy Comparison Table */}
      <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/90 p-4 shadow-xl sm:p-6">
        <h3 className="text-base font-bold text-white">
          Architectural Privacy Comparison Across Paradigms
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[750px] text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                <th className="px-3 py-3">Paradigm</th>
                <th className="px-3 py-3">Image Transmission</th>
                <th className="px-3 py-3">Compliance Considerations</th>
                <th className="px-3 py-3">Inversion Resistance</th>
                <th className="px-3 py-3">Implemented in App</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-800/60">
              {/* Centralized Learning */}
              <tr className="transition-colors hover:bg-slate-800/40">
                <td className="px-3 py-3 font-semibold text-rose-400">
                  Centralized Learning
                </td>
                <td className="px-3 py-3 text-rose-300">
                  Raw images sent to a central dataset
                </td>
                <td className="px-3 py-3 text-rose-300">
                  Greater centralized data exposure
                </td>
                <td className="px-3 py-3 text-rose-300">
                  Depends on security controls
                </td>
                <td className="px-3 py-3 text-slate-500">No</td>
              </tr>

              {/* Basic Federated Learning */}
              <tr className="border-l-4 border-cyan-400 bg-cyan-500/10 font-bold">
                <td className="px-3 py-3 text-cyan-300">
                  Basic Federated Learning (FedAvg)
                </td>
                <td className="px-3 py-3 text-emerald-400">
                  Raw images are not centrally pooled
                </td>
                <td className="px-3 py-3 text-cyan-300">
                  Reduced raw-data exposure
                </td>
                <td className="px-3 py-3 text-amber-300">
                  Not guaranteed
                </td>
                <td className="px-3 py-3 text-emerald-400">
                  Yes — Current System
                </td>
              </tr>

              {/* Differential Privacy */}
              <tr className="transition-colors hover:bg-slate-800/40">
                <td className="px-3 py-3 font-semibold text-purple-300">
                  DP-Federated Learning
                </td>
                <td className="px-3 py-3 text-emerald-400">
                  Raw images need not be centrally pooled
                </td>
                <td className="px-3 py-3 text-slate-300">
                  Depends on implementation and privacy budget
                </td>
                <td className="px-3 py-3 text-emerald-400">
                  Formal guarantees with correctly implemented DP
                </td>
                <td className="px-3 py-3 text-slate-500">
                  Future Scope
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};