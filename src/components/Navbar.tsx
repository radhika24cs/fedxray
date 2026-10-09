
import React from 'react';
import {
  Activity,
  Layers,
  Building2,
  BookOpen,
  BarChart3,
  ShieldCheck,
  Stethoscope,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
} from 'lucide-react';

export type ActiveTab =
  | 'dashboard'
  | 'prediction'
  | 'federated'
  | 'hospitals'
  | 'rag'
  | 'privacy';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  hasActivePrediction: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  hasActivePrediction,
}) => {
  const [collapsed, setCollapsed] = React.useState(false);
  const [mobileOpen, setMobileOpen] = React.useState(false);

  const navItems = [
    { id: 'dashboard' as ActiveTab, label: 'Dashboard', icon: Activity },
    {
      id: 'prediction' as ActiveTab,
      label: 'X-Ray Prediction',
      icon: Stethoscope,
      badge: hasActivePrediction,
    },
    { id: 'federated' as ActiveTab, label: 'Federated Learning', icon: Layers },
    { id: 'hospitals' as ActiveTab, label: 'Hospital Silos', icon: Building2 },
    { id: 'rag' as ActiveTab, label: 'Medical RAG', icon: BookOpen },
    { id: 'privacy' as ActiveTab, label: 'Privacy Protocol', icon: ShieldCheck },
  ];

  return (
    <>
      {/* Mobile menu button */}
      <button
        type="button"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
        className="fixed left-4 top-4 z-[70] flex h-10 w-10 items-center justify-center rounded-lg border border-slate-700 bg-slate-900 text-white shadow-lg lg:hidden"
      >
        {mobileOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Mobile backdrop */}
      {mobileOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
        />
      )}

      {/* Left sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex flex-col border-r border-slate-800 bg-slate-950 text-white transition-all duration-300 ${
          collapsed ? 'w-[76px]' : 'w-[260px]'
        } ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Logo */}
        <button
          type="button"
          onClick={() => {
            setActiveTab('dashboard');
            setMobileOpen(false);
          }}
          className={`flex h-[76px] shrink-0 items-center border-b border-slate-800 hover:bg-slate-900 ${
            collapsed ? 'justify-center px-2' : 'gap-3 px-5'
          }`}
        >
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 shadow-lg shadow-cyan-500/20">
            <Stethoscope className="h-5 w-5 text-white" />
          </div>

          {!collapsed && (
            <div className="text-left">
              <span className="block text-lg font-bold">FedX-Ray</span>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-cyan-400">
                FedAvg + RAG
              </span>
            </div>
          )}
        </button>

        {!collapsed && (
          <p className="px-5 pb-2 pt-6 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            Workspace
          </p>
        )}

        {/* Navigation options */}
        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;

            return (
              <button
                key={item.id}
                type="button"
                title={collapsed ? item.label : undefined}
                onClick={() => {
                  setActiveTab(item.id);
                  setMobileOpen(false);
                }}
                className={`group flex w-full items-center rounded-xl border py-3 transition-all ${
                  collapsed
                    ? 'justify-center px-2'
                    : 'justify-start gap-3 px-3'
                } ${
                  isActive
                    ? 'border-cyan-500/20 bg-cyan-500/10 text-cyan-300'
                    : 'border-transparent text-slate-400 hover:bg-slate-900 hover:text-white'
                }`}
              >
                <Icon
                  className={`h-5 w-5 shrink-0 ${
                    isActive ? 'text-cyan-400' : 'text-slate-400'
                  }`}
                />

                {!collapsed && (
                  <>
                    <span className="flex-1 text-left text-sm font-medium">
                      {item.label}
                    </span>
                    {item.badge && (
                      <span
                        className="h-2 w-2 rounded-full bg-emerald-400"
                        title="Active prediction"
                      />
                    )}
                  </>
                )}
              </button>
            );
          })}
        </nav>

        {/* Collapse / expand */}
        <div className="border-t border-slate-800 p-3">
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className="hidden w-full items-center justify-center gap-2 rounded-lg py-3 text-xs text-slate-400 hover:bg-slate-900 hover:text-white lg:flex"
          >
            {collapsed ? (
              <ChevronRight size={18} />
            ) : (
              <>
                <ChevronLeft size={18} />
                <span>Collapse sidebar</span>
              </>
            )}
          </button>

          {!collapsed && (
            <div className="mt-2 rounded-lg bg-slate-900 p-3">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-cyan-400" />
                <span className="text-xs font-medium">Privacy-first AI</span>
              </div>
              <p className="mt-1 text-[10px] leading-relaxed text-slate-500">
                Federated learning with medical insights.
              </p>
            </div>
          )}
        </div>
      </aside>
    </>
  );
};