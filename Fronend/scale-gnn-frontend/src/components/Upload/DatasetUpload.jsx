import React, { useState } from 'react';
import { Upload, CheckCircle, AlertCircle } from 'lucide-react';
import { API_BASE } from '../../api/api';

export default function DatasetUpload({ onSuccess }) {
  const [name, setName] = useState('');
  const [files, setFiles] = useState({
    coverage_edges: null,
    history: null,
    test_runs: null,
    changed_files: null,
  });

  const [uploading, setUploading] = useState(false);
  const [successMessage, setSuccessMessage] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleSubmit = async () => {
    if (!name || !Object.values(files).every(Boolean)) {
      setErrorMessage('Please provide dataset name and all required files.');
      return;
    }

    setUploading(true);
    setErrorMessage(null);
    setSuccessMessage(null);

    const formData = new FormData();
    formData.append('name', name);
    formData.append('coverage_edges', files.coverage_edges);
    formData.append('history', files.history);
    formData.append('test_runs', files.test_runs);
    formData.append('changed_files', files.changed_files);

    // 🔍 Debug helper (optional)
    // for (let pair of formData.entries()) {
    //   console.log(pair[0], pair[1]);
    // }

    try {
      const res = await fetch(`${API_BASE}/datasets/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.detail || 'Upload failed. Please try again.');
      }

      setSuccessMessage('Dataset uploaded successfully 🎉');
      setName('');
      setFiles({
        coverage_edges: null,
        history: null,
        test_runs: null,
        changed_files: null,
      });

      onSuccess?.();

    } catch (err) {
      console.error('Upload error:', err);
      setErrorMessage(err.message || 'Something went wrong during upload.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-slate-800 rounded-xl shadow-2xl p-8">
      <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
        <Upload className="w-6 h-6" />
        Upload Dataset
      </h2>

      <div className="space-y-6">

        {/* Success Message */}
        {successMessage && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-green-600/20 text-green-400 border border-green-600">
            <CheckCircle className="w-5 h-5" />
            {successMessage}
          </div>
        )}

        {/* Error Message */}
        {errorMessage && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-600/20 text-red-400 border border-red-600">
            <AlertCircle className="w-5 h-5" />
            {errorMessage}
          </div>
        )}

        {/* Dataset Name */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Dataset Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. GNN Test Prioritization v1"
            className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>

        {/* File Inputs */}
        {['coverage_edges', 'history', 'test_runs', 'changed_files'].map(field => (
          <div key={field}>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              {field
                .split('_')
                .map(w => w.charAt(0).toUpperCase() + w.slice(1))
                .join(' ')}
              {field !== 'changed_files' ? ' (.csv)' : ' (.txt)'}
            </label>
            <input
              type="file"
              accept={field === 'changed_files' ? '.txt' : '.csv'}
              onChange={(e) =>
                setFiles(prev => ({
                  ...prev,
                  [field]: e.target.files[0],
                }))
              }
              className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-300
                         file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0
                         file:bg-purple-600 file:text-white hover:file:bg-purple-700"
            />
          </div>
        ))}

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          disabled={uploading || !name || !Object.values(files).every(Boolean)}
          className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700
                     disabled:bg-slate-600 text-white rounded-lg font-medium
                     transition-colors flex items-center justify-center gap-2"
        >
          <Upload className="w-5 h-5" />
          {uploading ? 'Uploading...' : 'Upload Dataset'}
        </button>
      </div>
    </div>
  );
}
