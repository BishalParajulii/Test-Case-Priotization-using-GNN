import RankingCurve from "./RankingCurve";
import ReductionChart from "./ReductionChart";

const ExperimentDashboard = ({ results }) => {
  return (
    <div style={{ display: "grid", gap: "30px" }}>
      <RankingCurve data={results.ranking} />
      <ReductionChart reduction={results.reduction} />
    </div>
  );
};

export default ExperimentDashboard;
