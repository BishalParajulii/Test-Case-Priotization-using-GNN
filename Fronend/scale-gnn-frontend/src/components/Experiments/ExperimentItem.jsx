import React, { useState } from 'react';
import { Play, Download, TrendingUp, CheckCircle, Clock, XCircle } from 'lucide-react';
import ResultsPanel from './ResultsPanel'; // make sure the path is correct

export default function ExperimentItem({ exp, onRun, onDownload }) {
  const [showResults, setShowResults] = useState(false);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'COMPLETED':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'RUNNING':
        return <Clock className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'FAILED':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const toggleResults = () => setShowResults((prev) => !prev);

  return (
    <div className="bg-slate-700 rounded-lg p-4 hover:bg-slate-650 transition-colors mb-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {getStatusIcon(exp.status)}
          <div>
            <h3 className="text-lg font-semibold text-white">{exp.name}</h3>
            <p className="text-sm text-slate-400">
              Dataset ID: {exp.dataset_id} • Created: {new Date(exp.created_at).toLocaleDateString()}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {exp.status === 'PENDING' && (
            <button
              onClick={() => onRun(exp.id)}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              <Play className="w-4 h-4" />
              Run
            </button>
          )}
          {exp.status === 'COMPLETED' && (
            <>
              <button
                onClick={toggleResults}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <TrendingUp className="w-4 h-4" />
                {showResults ? 'Hide Results' : 'View Results'}
              </button>
              <button
                onClick={() => onDownload(exp.id)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <Download className="w-4 h-4" />
                Report
              </button>
            </>
          )}
        </div>
      </div>

      {/* ResultsPanel toggle */}
      {showResults && (
        <div className="mt-4">
          <ResultsPanel experimentId={exp.id} />
        </div>
      )}
    </div>
  );
}
