import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

const ReductionChart = ({ reduction }) => {
  if (!reduction) return null;

  const chartData = [
    {
      stage: "Original",
      tests: reduction.fault_prune?.[0]?.tests_before || 0,
    },
    {
      stage: "After Fault Prune",
      tests: reduction.fault_prune?.[0]?.tests_after || 0,
    },
    {
      stage: "After Spectral",
      tests: reduction.spectral?.[0]?.tests_after || 0,
    },
  ];

  return (
    <div>
      <h3>Test Reduction</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="stage" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="tests" fill="#10b981" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ReductionChart;
