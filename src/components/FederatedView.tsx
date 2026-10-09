
import React, { useEffect, useState } from 'react';
import {
  Layers,
  RefreshCw,
  CheckCircle2,
  Building2,
  ShieldCheck,
  Cpu,
  ArrowDown,
  ArrowUp,
  Server,
  TrendingUp,
} from 'lucide-react';

interface FederatedClient {
  local_loss: number;
  local_accuracy: number;
}

interface FederatedRound {
  round: number;
  clients: FederatedClient[];
  validation_accuracy: number;
  global_test_accuracy: number;
  global_test_f1: number;
}

export const FederatedView: React.FC = () => {
  const [rounds, setRounds] = useState<FederatedRound[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationMessage, setSimulationMessage] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/federated/metrics')
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load federated metrics');
        return r.json();
      })
      .then((data) => {
        if (data.rounds) {
          setRounds(data.rounds);
        }
      })
      .catch((err) => console.error('Failed to load federated metrics:', err));
  }, []);

  const triggerNextRound = async () => {
    setIsSimulating(true);
    setSimulationMessage(null);

    try {
      const res = await fetch('/api/federated/simulate-round', {
        method: 'POST',
      });

      if (!res.ok) {
        throw new Error('Failed to simulate the next round');
      }

      const data = await res.json();

      if (data.newRound) {
        setRounds((prev) => [...prev, data.newRound]);
        setSimulationMessage(
          data.message || `Federated round ${data.newRound.round} completed.`
        );
      } else {
        setSimulationMessage(
          data.message || 'The server did not return a new round.'
        );
      }
    } catch (err) {
      console.error('Simulate round error:', err);
      setSimulationMessage(
        'Unable to simulate the next round. Please check the server and try again.'
      );
    } finally {
      setIsSimulating(false);
    }
  };

  const latestRound = rounds[rounds.length - 1];

  const formatLoss = (value?: number) =>
    typeof value === 'number' && Number.isFinite(value)
      ? value.toFixed(3)
      : '--';

  const formatAccuracy = (value?: number) =>
    typeof value === 'number' && Number.isFinite(value)
      ? `${(value * 100).toFixed(1)}%`
      : '--';

  const formatPercent = (value?: number) =>
    typeof value === 'number' && Number.isFinite(value)
      ? `${value.toFixed(1)}%`
      : '--';

  const hospitals = [
    {
      name: 'Hospital A',
      specialty: 'Pulmonology & Acute Respiratory Care',
      description:
        'Non-IID Skew: High Pneumonia (31%) & Infiltration (24%). Minimal cardiomegaly.',
      count: '145 Radiographs',
      color: 'text-cyan-400',
      borderHover: 'hover:border-cyan-500/40',
      fallbackLoss: '1.4955',
      fallbackAccuracy: '34.5%',
    },
    {
      name: 'Hospital B',
      specialty: 'Cardiovascular & Thoracic Care Center',
      description:
        'Non-IID Skew: High Cardiomegaly (31%) & Pleural Effusion (28%). Minimal pneumonia.',
      count: '145 Radiographs',
      color: 'text-blue-400',
      borderHover: 'hover:border-blue-500/40',
      fallbackLoss: '1.0856',
      fallbackAccuracy: '63.4%',
    },
    {
      name: 'Hospital C',
      specialty: 'Metropolitan General Teaching Hospital',
      description:
        'Non-IID Skew: High Atelectasis (25%) & balanced acute cases. General metropolitan mix.',
      count: '160 Radiographs',
      color: 'text-indigo-400',
      borderHover: 'hover:border-indigo-500/40',
      fallbackLoss: '1.4651',
      fallbackAccuracy: '42.5%',
    },
  ];

  return (
    <div className="w-full min-w-0 space-y-6 overflow-x-hidden p-4 sm:p-6 lg:p-8">
      {/* Page header */}
      <header className="min-w-0 border-b border-slate-800/80 pb-6">
        <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">
          <Layers className="h-4 w-4 shrink-0" />
          <span>Decentralized Architecture</span>
        </div>

        <h2 className="break-words text-2xl font-extrabold leading-tight tracking-tight text-white sm:text-3xl">
          Federated Learning &amp; FedAvg Aggregation
        </h2>

        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400 sm:text-base">
          Simulated collaboration across Hospital A, Hospital B, and Hospital C.
          Each hospital keeps its raw patient images locally while model updates
          are combined through Federated Averaging.
        </p>
      </header>

      {/* Simulation feedback */}
      {simulationMessage && (
        <div
          role="status"
          className="flex min-w-0 items-start gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4 text-sm text-emerald-300"
        >
          <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0" />
          <span className="break-words">{simulationMessage}</span>
        </div>
      )}

      {/* Federated architecture */}
      <section className="relative min-w-0 space-y-7 overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950 p-4 shadow-xl sm:space-y-8 sm:p-6 lg:p-8">
        <div className="space-y-2 text-center">
          <span className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-400">
            Live Federated Topology
          </span>
          <h3 className="mx-auto max-w-3xl text-base font-bold leading-relaxed text-white sm:text-lg">
            Distributed Local Training &amp; Centralized Parameter Averaging
          </h3>
          <p className="mx-auto max-w-2xl text-xs leading-5 text-slate-400 sm:text-sm">
            Global model distribution, local hospital training, and aggregated
            model updates.
          </p>
        </div>

        {/* Global model */}
        <div className="flex min-w-0 justify-center">
          <div className="w-full max-w-xl min-w-0 space-y-3 rounded-xl border border-blue-500/40 bg-gradient-to-r from-blue-900/60 to-indigo-900/60 p-4 text-center shadow-lg shadow-blue-500/10 sm:p-5">
            <div className="inline-flex max-w-full flex-wrap items-center justify-center gap-2 rounded-full bg-blue-500/20 px-3 py-1 text-[11px] font-bold text-blue-300">
              <Cpu className="h-4 w-4 shrink-0" />
              <span>CENTRAL GLOBAL MODEL (FedChestNet)</span>
            </div>

            <p className="text-xs leading-5 text-slate-300 sm:text-sm">
              Dispatches current model weights W_global to participating
              hospital nodes.
            </p>

            <div className="flex flex-wrap justify-center gap-x-4 gap-y-2 font-mono text-xs text-cyan-300">
              <span>Rounds: {rounds.length}</span>
              <span>
                Test Acc: {latestRound ? formatPercent(latestRound.global_test_accuracy) : '--'}
              </span>
              <span>
                Test F1: {latestRound ? formatPercent(latestRound.global_test_f1) : '--'}
              </span>
            </div>
          </div>
        </div>

        {/* Broadcast arrows */}
        <div className="grid grid-cols-3 gap-2 text-center">
          {['Hospital A', 'Hospital B', 'Hospital C'].map((name) => (
            <div key={name} className="flex min-w-0 flex-col items-center gap-2">
              <ArrowDown className="h-5 w-5 animate-bounce text-blue-400" />
              <span className="text-[10px] leading-4 text-blue-300 sm:text-xs">
                Send Weights W_t
              </span>
            </div>
          ))}
        </div>

        {/* Hospital cards */}
        
<div className="grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-3 lg:gap-5 lg:ml-6">
          {hospitals.map((hospital, index) => {
            const client = latestRound?.clients?.[index];

            return (
              <article
                key={hospital.name}
                className={`flex min-w-0 flex-col gap-4 rounded-xl border border-slate-700/80 bg-slate-950/70 p-4 transition-colors sm:p-5 ${hospital.borderHover}`}
              >
                <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                  <span className={`flex items-center gap-2 text-sm font-bold ${hospital.color}`}>
                    <Building2 className="h-4 w-4 shrink-0" />
                    {hospital.name}
                  </span>

                  <span className="rounded-md bg-slate-800 px-2 py-1 text-[10px] text-slate-300">
                    {hospital.count}
                  </span>
                </div>

                <h4 className="text-sm font-semibold leading-5 text-slate-200">
                  {hospital.specialty}
                </h4>

                <p className="text-xs leading-5 text-slate-400">
                  {hospital.description}
                </p>

                <div className="mt-auto space-y-3 rounded-lg border border-slate-800 bg-slate-900/80 p-3">
                  <div className="flex items-start justify-between gap-3 text-xs">
                    <span className="text-slate-400">Local Training Loss</span>
                    <span className={`font-mono font-semibold ${hospital.color}`}>
                      {client && Number.isFinite(client.local_loss)
                        ? client.local_loss.toFixed(4)
                        : hospital.fallbackLoss}
                    </span>
                  </div>

                  <div className="flex items-start justify-between gap-3 text-xs">
                    <span className="text-slate-400">Local Accuracy</span>
                    <span className={`font-mono font-semibold ${hospital.color}`}>
                      {client && Number.isFinite(client.local_accuracy)
                        ? formatAccuracy(client.local_accuracy)
                        : hospital.fallbackAccuracy}
                    </span>
                  </div>
                </div>

                <div className="flex items-start gap-2 border-t border-slate-800 pt-3 text-[10px] font-medium leading-4 text-emerald-400">
                  <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>Private Patient Data Stays On-Premise</span>
                </div>
              </article>
            );
          })}
        </div>

        {/* Model update arrows */}
        <div className="grid grid-cols-3 gap-2 text-center">
          {[
            { label: 'Weights Update ΔW_A', color: 'text-cyan-400' },
            { label: 'Weights Update ΔW_B', color: 'text-blue-400' },
            { label: 'Weights Update ΔW_C', color: 'text-indigo-400' },
          ].map((item) => (
            <div key={item.label} className="flex min-w-0 flex-col items-center gap-2">
              <span className={`text-[10px] leading-4 sm:text-xs ${item.color}`}>
                {item.label}
              </span>
              <ArrowUp className={`h-5 w-5 animate-bounce ${item.color}`} />
            </div>
          ))}
        </div>

        {/* FedAvg server */}
        <div className="flex min-w-0 justify-center">
          <div className="w-full max-w-2xl min-w-0 space-y-3 rounded-2xl border border-cyan-500/40 bg-gradient-to-r from-cyan-950/70 to-blue-950/70 p-4 text-center shadow-xl sm:p-5">
            <div className="inline-flex max-w-full flex-wrap items-center justify-center gap-2 rounded-full bg-cyan-500/20 px-3 py-1 text-xs font-bold text-cyan-300">
              <Server className="h-4 w-4 shrink-0" />
              <span>FEDERATED SERVER (FedAvg Aggregation Engine)</span>
            </div>

            <div className="break-words rounded-lg border border-cyan-800/40 bg-slate-900/80 p-3 font-mono text-[10px] leading-5 text-cyan-200 sm:text-xs">
              W_(t+1) = (145/450)·W_A + (145/450)·W_B + (160/450)·W_C
            </div>

            <p className="text-xs leading-5 text-slate-300 sm:text-sm">
              Performs weighted averaging of parameters based on sample counts.
              Patient images are not transmitted to the server.
            </p>
          </div>
        </div>
      </section>

      {/* Round history */}
      <section className="min-w-0 space-y-4 overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/80 p-4 shadow-xl sm:p-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="flex items-center gap-2 text-base font-bold text-white">
            <TrendingUp className="h-5 w-5 shrink-0 text-cyan-400" />
            <span>Federated Convergence Trajectory</span>
          </h3>

          <span className="text-xs text-slate-400">
            {rounds.length} Total Rounds Completed
          </span>
        </div>

        {rounds.length === 0 ? (
          <p className="rounded-lg border border-slate-800 bg-slate-950/50 p-4 text-sm text-slate-400">
            No round history has been returned by the server yet. You can
            simulate a round using the button below.
          </p>
        ) : (
          <div className="w-full min-w-0 overflow-x-auto">
            <table className="w-full min-w-[850px] text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-400 sm:text-[11px]">
                  <th className="px-3 py-3">Round</th>
                  <th className="px-3 py-3">Hospital A (Loss / Acc)</th>
                  <th className="px-3 py-3">Hospital B (Loss / Acc)</th>
                  <th className="px-3 py-3">Hospital C (Loss / Acc)</th>
                  <th className="px-3 py-3 text-cyan-400">Val Accuracy</th>
                  <th className="px-3 py-3 text-cyan-400">Global Test Acc</th>
                  <th className="px-3 py-3 text-cyan-400">Global Test F1</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-800/60 font-mono">
                {rounds.map((round) => (
                  <tr
                    key={round.round}
                    className="transition-colors hover:bg-slate-800/40"
                  >
                    <td className="whitespace-nowrap px-3 py-3 font-bold text-white">
                      Round {round.round}
                    </td>

                    {[0, 1, 2].map((index) => (
                      <td key={index} className="whitespace-nowrap px-3 py-3 text-slate-300">
                        {formatLoss(round.clients?.[index]?.local_loss)} /{' '}
                        {formatAccuracy(round.clients?.[index]?.local_accuracy)}
                      </td>
                    ))}

                    <td className="whitespace-nowrap px-3 py-3 font-bold text-slate-200">
                      {formatPercent(round.validation_accuracy)}
                    </td>

                    <td className="whitespace-nowrap px-3 py-3 font-bold text-cyan-400">
                      {formatPercent(round.global_test_accuracy)}
                    </td>

                    <td className="whitespace-nowrap px-3 py-3 font-bold text-cyan-300">
                      {formatPercent(round.global_test_f1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Simulate next round - bottom of page */}
      <div className="flex flex-col gap-3 border-t border-slate-800/80 pt-6 sm:flex-row sm:items-center sm:justify-between">
        <p className="max-w-xl text-xs leading-5 text-slate-500">
          Run another federated training round to request updated metrics from
          the server.
        </p>

        <button
          type="button"
          onClick={triggerNextRound}
          disabled={isSimulating}
          className="inline-flex w-full shrink-0 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-cyan-500/20 transition-all hover:from-cyan-400 hover:to-blue-500 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
        >
          <RefreshCw
            className={`h-4 w-4 ${isSimulating ? 'animate-spin' : ''}`}
          />
          <span>
            {isSimulating ? 'Executing Federated Round...' : 'Simulate Next Round'}
          </span>
        </button>
      </div>
    </div>
  );
};