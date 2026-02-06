import React from "react"

export default function TopTests({ data }) {
  if (!data || data.length === 0) {
    return <p className="text-slate-400">No top test data available</p>
  }

  return (
    <div className="bg-slate-700 rounded-lg p-6 overflow-x-auto">
      <h3 className="text-lg font-semibold text-white mb-4">Top 10 Tests</h3>
      <table className="w-full text-sm text-slate-300 border-collapse">
        <thead className="text-xs uppercase bg-slate-600 text-slate-300">
          <tr>
            <th className="px-4 py-2 text-left">Rank</th>
            <th className="px-4 py-2 text-left">Test ID</th>
            <th className="px-4 py-2 text-left">Cluster ID</th>
            <th className="px-4 py-2 text-left">Risk Score</th>
          </tr>
        </thead>
        <tbody>
          {data.map((test, index) => (
            <tr key={index} className="border-b border-slate-600">
              <td className="px-4 py-2">{index + 1}</td>
              <td className="px-4 py-2 font-mono">{test.test_id}</td>
              <td className="px-4 py-2">{test.cluster_test_id ?? 'N/A'}</td>
              <td className="px-4 py-2">{test.risk_score?.toFixed(6) ?? 'N/A'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
