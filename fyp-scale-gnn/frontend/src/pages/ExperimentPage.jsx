import { useEffect, useState } from "react";
import { getExperimentResults } from "../api/results";

import RankingCurve from "../components/RankingCurve";
import TopTests from "../components/TopTests";
import Metrics from "../components/Metrics";
import ReportButton from "../components/ReportButton";

const ExperimentPage = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getExperimentResults(1)
      .then((res) => {
        console.log("API RESPONSE:", res.data);
        setData(res.data);
      })
      .catch((err) => {
        console.error(err);
        setError("Failed to load experiment");
      });
  }, []);

  if (error) {
    return (
      <div style={{ padding: "30px", color: "red" }}>
        {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ padding: "30px" }}>
        <h3>Loading experiment...</h3>
      </div>
    );
  }

  return (
    <div
      style={{
        padding: "30px",
        background: "#f8fafc",
        minHeight: "100vh",
        color: "#0f172a",
      }}
    >
      <h2 style={{ marginBottom: "10px" }}>
        Experiment #{data.experiment_id}
      </h2>

      {/* 📈 Ranking Curve */}
      {data.ranking && data.ranking.length > 0 ? (
        <RankingCurve data={data.ranking} />
      ) : (
        <p>No ranking data available</p>
      )}

      {/* 🧪 Top-K Tests */}
      <TopTests tests={data.top_tests} />

      {/* 📊 Metrics */}
      <Metrics metrics={data.metrics} />

      {/* 📄 PDF Report */}
      <ReportButton experimentId={data.experiment_id} />
    </div>
  );
};

export default ExperimentPage;
