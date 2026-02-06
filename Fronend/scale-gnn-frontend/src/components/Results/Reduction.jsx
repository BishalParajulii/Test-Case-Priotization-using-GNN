import React from "react";

const Reduction = ({ data }) => {
  if (!data) return <p className="text-slate-300">No reduction data found.</p>;

  // Format strategy_params nicely
  const formatParams = (paramStr) => {
    if (!paramStr) return "";
    return paramStr
      .replace(/[\{\}]/g, "") // remove curly braces
      .replace(/'/g, "")      // remove single quotes
      .replace(/,/g, ", ");   // add space after commas
  };

  const renderTable = (title, items) => (
    <div className="bg-slate-700 rounded-lg p-4 mb-6">
      <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
      {items && items.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-slate-300">
            <thead className="text-xs uppercase bg-slate-600 text-slate-300">
              <tr>
                <th className="px-3 py-2 text-left">Strategy</th>
                <th className="px-3 py-2 text-left">Params</th>
                <th className="px-3 py-2 text-left">Original Tests</th>
                <th className="px-3 py-2 text-left">Reduced Tests</th>
                <th className="px-3 py-2 text-left">Test Retention</th>
                <th className="px-3 py-2 text-left">Original Edges</th>
                <th className="px-3 py-2 text-left">Reduced Edges</th>
                <th className="px-3 py-2 text-left">Edge Retention</th>
                <th className="px-3 py-2 text-left">Jaccard Fidelity</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item, idx) => (
                <tr key={idx} className="border-b border-slate-600">
                  <td className="px-3 py-2">{item.strategy_name}</td>
                  <td className="px-3 py-2 font-mono text-green-300">
                    {formatParams(item.strategy_params)}
                  </td>
                  <td className="px-3 py-2">{item.orig_tests}</td>
                  <td className="px-3 py-2">{item.red_tests}</td>
                  <td className="px-3 py-2">{item.tests_retention_ratio.toFixed(4)}</td>
                  <td className="px-3 py-2">{item.orig_edges}</td>
                  <td className="px-3 py-2">{item.red_edges}</td>
                  <td className="px-3 py-2">{item.edges_retention_ratio.toFixed(4)}</td>
                  <td className="px-3 py-2">{item.jaccard_fidelity_mean.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-slate-400">No data available</p>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      {renderTable("Fault Prune Strategies", data.fault_prune)}
      {renderTable("Spectral Strategies", data.spectral)}
    </div>
  );
};

export default Reduction;
