import React from 'react';
import ExperimentItem from './ExperimentItem';
import { FileText } from 'lucide-react';

export default function ExperimentList({ experiments, onRun, onViewResults, onDownload }) {
  if (experiments.length === 0) {
    return (
      <div className="text-center py-12 text-slate-400">
        <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
        <p>No experiments yet. Create one to get started!</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {experiments.map((exp) => (
        <ExperimentItem
          key={exp.id}
          exp={exp}
          onRun={onRun}
          onViewResults={onViewResults}
          onDownload={onDownload}
        />
      ))}
    </div>
  );
}
