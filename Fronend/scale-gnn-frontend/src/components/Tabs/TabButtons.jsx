import React from 'react';

export default function TabButtons({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'experiments', label: 'Experiments' },
    { id: 'upload', label: 'Upload Dataset' },
  ];

  return (
    <div className="flex gap-4 mb-6">
      {tabs.map(tab => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={`px-6 py-3 rounded-lg font-medium transition-all ${
            activeTab === tab.id
              ? 'bg-purple-600 text-white shadow-lg'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
