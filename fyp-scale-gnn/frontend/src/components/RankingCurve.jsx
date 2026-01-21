import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

const RankingCurve = ({ data }) => {
  if (!Array.isArray(data) || data.length === 0) {
    return <p>No ranking data available</p>;
  }

  // ✅ Safely parse risk scores
  const risks = data
    .map((d) => Number(d.risk_score))
    .filter((v) => !isNaN(v) && isFinite(v));

  if (risks.length === 0) {
    return <p>Invalid risk score data</p>;
  }

  const maxRisk = Math.max(...risks);

  const chartData = data
    .map((row) => {
      const risk = Number(row.risk_score);
      if (isNaN(risk)) return null;

      return {
        rank: Number(row.rank),
        risk: risk / maxRisk,
      };
    })
    .filter(Boolean);

  return (
    <div
      style={{
        background: "#ffffff",
        padding: "20px",
        borderRadius: "8px",
        boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
      }}
    >
      <h3>Test Prioritization Curve (Normalized Risk)</h3>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={chartData}>
          <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />

          <XAxis
            dataKey="rank"
            label={{
              value: "Test Execution Order",
              position: "insideBottom",
              offset: -5,
            }}
          />

          <YAxis
            domain={[0, 1]}
            label={{
              value: "Normalized Risk Score",
              angle: -90,
              position: "insideLeft",
            }}
          />

          <Tooltip formatter={(v) => v.toFixed(4)} />

          <Line
            type="monotone"
            dataKey="risk"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RankingCurve;
