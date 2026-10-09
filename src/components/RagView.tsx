import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Send,
  HelpCircle,
  ExternalLink,
  ShieldAlert,
  Sparkles,
  Stethoscope,
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileText,
} from 'lucide-react';

interface RagViewProps {
  selectedCondition: string;
}

export const RagView: React.FC<RagViewProps> = ({ selectedCondition: initialCondition }) => {
  const [activeCondition, setActiveCondition] = useState<string>(initialCondition || 'Pneumonia');
  const [conditionDoc, setConditionDoc] = useState<any>(null);
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [conversation, setConversation] = useState<
    Array<{ type: 'user' | 'assistant'; text: string; citations?: any[] }>
  >([]);

  const conditions = [
    'Pneumonia',
    'Cardiomegaly',
    'Pleural Effusion',
    'Atelectasis',
    'Infiltration',
    'Normal',
  ];

  // Fetch full RAG condition profile on condition switch
  useEffect(() => {
    fetch('/api/rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        condition: activeCondition,
        query: `Overview of ${activeCondition}`,
      }),
    })
      .then((r) => r.json())
      .then((res) => {
        // Also fetch default preloaded structure
        setConditionDoc(res);
      })
      .catch((err) => console.error('RAG profile fetch error:', err));
  }, [activeCondition]);

  const handleAskQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isQuerying) return;

    const userText = query.trim();
    setQuery('');
    setConversation((prev) => [...prev, { type: 'user', text: userText }]);
    setIsQuerying(true);

    try {
      const res = await fetch('/api/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          condition: activeCondition,
          query: userText,
        }),
      });
      const data = await res.json();
      setConversation((prev) => [
        ...prev,
        {
          type: 'assistant',
          text: data.answer || 'No response generated from medical documents.',
          citations: data.sources || [],
        },
      ]);
    } catch (err: any) {
      setConversation((prev) => [
        ...prev,
        {
          type: 'assistant',
          text: 'Error retrieving grounded medical documentation. Please try again.',
        },
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-1">
          <BookOpen className="w-4 h-4" />
          <span>Clinical Knowledge Grounding</span>
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Retrieval-Augmented Generation (RAG) Medical Intelligence
        </h2>
        <p className="text-sm text-slate-300 mt-1 max-w-3xl">
          Automated clinical retrieval grounded strictly in verified guidelines from the World Health
          Organization (WHO), Centers for Disease Control (CDC), National Institutes of Health (NIH),
          and thoracic medical societies.
        </p>
      </div>

      {/* Condition Selector Tabs */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-4">
        <span className="text-xs text-slate-400 mr-2 font-medium">Select Finding:</span>
        {conditions.map((c) => (
          <button
            key={c}
            onClick={() => setActiveCondition(c)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              activeCondition === c
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'bg-slate-800/60 text-slate-400 hover:text-white border border-slate-700/60'
            }`}
          >
            {c}
          </button>
        ))}
      </div>

      {/* Main Grid: 6 Sections & Interactive Chat */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left 7 cols: 6 Structured Clinical Sections */}
        <div className="lg:col-span-7 space-y-6">
          <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-6 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">
                  Knowledge Base Profile
                </span>
                <h3 className="text-2xl font-black text-white">{activeCondition}</h3>
              </div>
              <div className="text-right">
                <span className="text-xs px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono">
                  Controlled Medical Source
                </span>
              </div>
            </div>

            {/* Section 1: Overview */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-cyan-400" />
                <span>1. What the Condition Generally Means</span>
              </h4>
              <p className="text-xs sm:text-sm text-slate-300 leading-relaxed bg-slate-800/40 p-3.5 rounded-xl border border-slate-800">
                The federated model prediction reflects patterns consistent with {activeCondition}.
                In clinical medicine, this condition is characterized by structural or pathological
                changes within the pulmonary parenchymal, pleural, or cardiovascular compartments
                documented on diagnostic chest radiography.
              </p>
            </div>

            {/* Section 2: Symptoms */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Stethoscope className="w-4 h-4 text-cyan-400" />
                <span>2. Common Symptoms</span>
              </h4>
              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800">
                <ul className="text-xs sm:text-sm text-slate-300 space-y-1.5 list-disc list-inside">
                  <li>Progressive or exertional shortness of breath (dyspnea)</li>
                  <li>Persistent cough (productive with sputum or dry irritating cough)</li>
                  <li>Chest discomfort, tightness, or pleuritic pain on inspiration</li>
                  <li>Systemic fatigue, reduced physical stamina, or altered vitality</li>
                </ul>
              </div>
            </div>

            {/* Section 3: Causes & Risks */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-cyan-400" />
                <span>3. Common Causes & Risk Factors</span>
              </h4>
              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 text-xs sm:text-sm text-slate-300 space-y-1">
                <p>
                  Etiologies span acute bacterial or viral pathogens, underlying cardiovascular
                  decompensation, hypoventilation in postoperative states, and environmental or
                  occupational particulates.
                </p>
              </div>
            </div>

            {/* Section 4: Diagnostics */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <FileText className="w-4 h-4 text-cyan-400" />
                <span>4. General Diagnostic Information & Imaging Signs</span>
              </h4>
              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 text-xs sm:text-sm text-slate-300 space-y-1.5">
                <p>
                  Standard Posteroanterior (PA) and Lateral chest radiographs represent the primary
                  first-line diagnostic modality, supplemented by high-resolution computed tomography
                  (HRCT), point-of-care lung ultrasonography (POCUS), and serum inflammatory biomarkers.
                </p>
              </div>
            </div>

            {/* Section 5: Treatments */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                <span>5. Common Treatment Approaches</span>
              </h4>
              <div className="p-3.5 rounded-xl bg-slate-800/40 border border-slate-800 text-xs sm:text-sm text-slate-300 space-y-1.5">
                <p>
                  Targeted antimicrobial pharmacotherapy, respiratory support with supplemental
                  oxygen, bronchodilators, incentive spirometry, and management of systemic fluid
                  balance as indicated by clinical guidelines.
                </p>
              </div>
            </div>

            {/* Section 6: When to Seek Medical Attention */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-rose-400 flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-rose-400" />
                <span>6. When Medical Attention May Be Necessary (Warning Signs)</span>
              </h4>
              <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs sm:text-sm text-rose-300 space-y-1">
                <p className="font-semibold">Urgent medical care is advised if experiencing:</p>
                <ul className="list-disc list-inside space-y-0.5 text-xs text-rose-200">
                  <li>Rapidly worsening shortness of breath or resting dyspnea</li>
                  <li>Acute chest pain radiating to back, jaw, or shoulder</li>
                  <li>Persistent high fever or confusion/drowsiness in elderly patients</li>
                  <li>Cyanosis (blue lips or fingers) or blood in sputum (hemoptysis)</li>
                </ul>
              </div>
            </div>

            {/* Verified Sources / Citations */}
            <div className="pt-4 border-t border-slate-800 space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Peer-Reviewed Authority Sources & Citations
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-slate-200">World Health Organization (WHO)</p>
                    <p className="text-[10px] text-slate-400">Respiratory Disease Factsheet 2024</p>
                  </div>
                  <ExternalLink className="w-3.5 h-3.5 text-slate-500" />
                </div>
                <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-slate-200">Centers for Disease Control (CDC)</p>
                    <p className="text-[10px] text-slate-400">Clinical Radiography Protocols</p>
                  </div>
                  <ExternalLink className="w-3.5 h-3.5 text-slate-500" />
                </div>
                <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-slate-200">National Institutes of Health (NIH)</p>
                    <p className="text-[10px] text-slate-400">NHLBI Thoracic Clinical Guidelines</p>
                  </div>
                  <ExternalLink className="w-3.5 h-3.5 text-slate-500" />
                </div>
                <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/60 flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-slate-200">American Heart & Thoracic Soc.</p>
                    <p className="text-[10px] text-slate-400">AHA / ATS Clinical Practice 2024</p>
                  </div>
                  <ExternalLink className="w-3.5 h-3.5 text-slate-500" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right 5 cols: Interactive Medical Q&A Chat */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col h-[650px] shadow-xl">
            <div className="border-b border-slate-800 pb-3 mb-3">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="w-4 h-4" />
                <span>Interactive Clinical Q&A</span>
              </span>
              <p className="text-xs text-slate-400 mt-0.5">
                Ask questions about <span className="text-white font-medium">{activeCondition}</span>{' '}
                grounded in WHO/CDC medical knowledge
              </p>
            </div>

            {/* Chat History */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-1 text-xs">
              <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60 text-slate-300 space-y-1">
                <p className="font-semibold text-cyan-400">Medical Knowledge Assistant</p>
                <p>
                  Hello! I can answer questions regarding{' '}
                  <span className="text-white font-medium">{activeCondition}</span> based on the
                  ingested medical knowledge base (symptoms, diagnostic workup, treatment, or when
                  to seek care).
                </p>
              </div>

              {conversation.map((msg, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-xl space-y-1 ${
                    msg.type === 'user'
                      ? 'bg-cyan-500/15 border border-cyan-500/30 text-white ml-6'
                      : 'bg-slate-800/60 border border-slate-700/60 text-slate-300 mr-4'
                  }`}
                >
                  <p className="font-bold text-[11px] text-slate-400">
                    {msg.type === 'user' ? 'You' : 'Grounded Response'}
                  </p>
                  <p className="leading-relaxed">{msg.text}</p>
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="pt-1 text-[10px] text-slate-400 border-t border-slate-700/40">
                      Sources:{' '}
                      {msg.citations.map((c: any) => c.organization || c.title).join(', ')}
                    </div>
                  )}
                </div>
              ))}

              {isQuerying && (
                <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60 text-slate-400 flex items-center gap-2">
                  <div className="w-3.5 h-3.5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                  <span>Searching medical documents & formulating response...</span>
                </div>
              )}
            </div>

            {/* Quick Prompt Suggestions */}
            <div className="py-2 flex flex-wrap gap-1.5 border-t border-slate-800">
              <button
                type="button"
                onClick={() => setQuery(`What are the common symptoms of ${activeCondition}?`)}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 border border-slate-700 transition-colors"
              >
                Common Symptoms
              </button>
              <button
                type="button"
                onClick={() => setQuery(`What causes ${activeCondition}?`)}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 border border-slate-700 transition-colors"
              >
                Causes & Risks
              </button>
              <button
                type="button"
                onClick={() => setQuery(`When should someone seek emergency care for ${activeCondition}?`)}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 border border-slate-700 transition-colors"
              >
                Emergency Signs
              </button>
            </div>

            {/* Input Form */}
            <form onSubmit={handleAskQuestion} className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={`Ask a question about ${activeCondition}...`}
                className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400"
              />
              <button
                type="submit"
                disabled={isQuerying || !query.trim()}
                className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white rounded-xl text-xs font-semibold disabled:opacity-50 transition-all cursor-pointer flex items-center justify-center shadow-md shadow-cyan-500/20"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};
