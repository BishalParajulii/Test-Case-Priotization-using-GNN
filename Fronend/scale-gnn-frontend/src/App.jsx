import React, { useState, useEffect } from 'react';
import Header from './components/Layout/Header';
import TabButtons from './components/Tabs/TabButtons';
import ExperimentList from './components/Experiments/ExperimentList';
import ResultsPanel from './components/Experiments/ResultsPanel';
import DatasetUpload from './components/Upload/DatasetUpload';
import ExperimentCreate from './components/Create/ExperimentCreate';
import { API_BASE, fetchJSON } from './api/api';

function App() {
  const [activeTab, setActiveTab] = useState('experiments');
  const [experiments, setExperiments] = useState([]);
  const [results, setResults] = useState(null);
  const [selectedExperiment, setSelectedExperiment] = useState(null);

  useEffect(() => { loadExperiments(); }, []);

  const loadExperiments = async () => {
    try {
      const data = await fetchJSON('/experiments/');
      setExperiments(data);
    } catch (err) {
      console.error('Failed to load experiments:', err);
    }
  };

  const runExperiment = async (id) => {
    try {
      await fetch(`${API_BASE}/experiments/${id}/run`, { method: 'POST' });
      alert('Experiment started!');
      loadExperiments();
    } catch (err) {
      console.error(err);
      alert('Failed to start experiment');
    }
  };

  const loadResults = async (id) => {
    try {
      const data = await fetchJSON(`/results/${id}`);
      setResults(data);
      setSelectedExperiment(id);
    } catch (err) {
      console.error(err);
      alert('Failed to load results');
    }
  };

  const downloadReport = (id) => {
    window.open(`${API_BASE}/reports/${id}`, '_blank');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="max-w-7xl mx-auto p-6">
        <Header />
        <TabButtons activeTab={activeTab} setActiveTab={setActiveTab} />

        {activeTab === 'experiments' && (
          <>
            <ExperimentList
              experiments={experiments}
              onRun={runExperiment}
              onViewResults={loadResults}
              onDownload={downloadReport}
            />
            {results && selectedExperiment && (
              <ResultsPanel results={results} experimentId={selectedExperiment} />
            )}
          </>
        )}

        {activeTab === 'upload' && <DatasetUpload onSuccess={() => setActiveTab('create')} />}
        {activeTab === 'create' && (
          <ExperimentCreate onSuccess={() => { loadExperiments(); setActiveTab('experiments'); }} />
        )}
      </div>
    </div>
  );
}

export default App;
