const Metrics = ({ metrics }) => {
  if (!metrics) return null;

  return (
    <div style={{ marginTop: "20px" }}>
      <h3>Evaluation Metrics</h3>
      <ul>
        {Object.entries(metrics).map(([k, v]) => (
          <li key={k}>
            <b>{k}:</b> {v}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Metrics;
