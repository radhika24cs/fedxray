
import React, { useState, useEffect } from 'react';

import { Navbar, ActiveTab } from './components/Navbar';
import { DashboardView } from './components/DashboardView';
import { PredictionView } from './components/PredictionView';
import { FederatedView } from './components/FederatedView';
import { HospitalsView } from './components/HospitalsView';
import { RagView } from './components/RagView';
import { PerformanceView } from './components/PerformanceView';
import { PrivacyView } from './components/PrivacyView';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');
  const [selectedConditionForRag, setSelectedConditionForRag] =
    useState<string>('Pneumonia');
  const [hasActivePrediction, setHasActivePrediction] =
    useState<boolean>(false);

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .catch((error) => console.error('Health check failed:', error));
  }, []);

  const handleSelectConditionForRag = (condition: string) => {
    setSelectedConditionForRag(condition);
    setHasActivePrediction(true);
  };

  return (
    <div className="min-h-screen w-full overflow-x-hidden bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-white">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasActivePrediction={hasActivePrediction}
      />

      {/* Reserve space for the fixed desktop sidebar */}
      <div className="min-h-screen min-w-0 lg:pl-[260px]">
        <main className="min-w-0 w-full px-4 py-8 sm:px-6 lg:px-8">
          {activeTab === 'dashboard' && (
            <DashboardView
              onNavigate={setActiveTab}
              hospitalsSummary={null}
              modelInfo={null}
            />
          )}

          {activeTab === 'prediction' && (
            <PredictionView
              onSelectConditionForRag={handleSelectConditionForRag}
              onNavigate={setActiveTab}
            />
          )}

          {activeTab === 'federated' && <FederatedView />}

          {activeTab === 'hospitals' && <HospitalsView />}

          {activeTab === 'rag' && (
            <RagView selectedCondition={selectedConditionForRag} />
          )}

          {activeTab === 'privacy' && <PrivacyView />}
        </main>

        <footer className="border-t border-slate-900 bg-slate-950/80 px-4 py-6 text-xs text-slate-500 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center justify-between gap-4 xl:flex-row">
            <div className="flex flex-wrap items-center justify-center gap-2">
              <span className="font-semibold text-slate-300">
                FedChestXray Research System
              </span>
              <span>•</span>
              <span>
                Privacy-Preserving Federated Learning &amp; RAG Medical Retrieval
              </span>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-4 text-slate-400">
              <span>NIH ChestX-ray14 &amp; CheXpert Benchmark Protocol</span>
              <span>•</span>
              <span>College / IEEE Demonstration</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}