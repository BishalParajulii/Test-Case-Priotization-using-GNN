import React from "react"

export default function Metrics({ data }) {
  if (!data) {
    return <p className="text-slate-400">No metrics available</p>
  }

  // Calculate F1 Score (avoid divide by zero)
  const precision = data.precision_at_k ?? 0
  const recall = data.recall_at_k ?? 0
  const f1_score = (precision + recall) === 0 ? 0 : (2 * precision * recall) / (precision + recall)

  return (
    <div className="bg-slate-700 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Metrics</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-slate-300">

        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="font-medium">APFD</p>
          <p className="font-bold text-white">{data.apfd.toFixed(4)}</p>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="font-medium">Delta APFD</p>
          <p className="font-bold text-white">{data.delta_apfd.toFixed(4)}</p>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="font-medium">Fault Retention Ratio</p>
          <p className="font-bold text-white">{data.fault_retention_ratio}</p>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="font-medium">Tests Ranked</p>
          <p className="font-bold text-white">{data.num_tests_ranked}</p>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="font-medium">Failures Current</p>
          <p className="font-bold text-white">{data.num_failures_current}</p>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="font-medium">Train Time (s)</p>
          <p className="font-bold text-white">{data.train_time_s.toFixed(3)}</p>
        </div>
      </div>
    </div>
  )
}
