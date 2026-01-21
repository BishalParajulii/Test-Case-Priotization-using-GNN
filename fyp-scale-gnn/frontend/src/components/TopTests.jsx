const TopTests = ({ tests }) => {
  if (!tests || tests.length === 0) return null;

  return (
    <div style={{ marginTop: "20px" }}>
      <h3>Top-K Prioritized Tests</h3>
      <ol>
        {tests.map((t, i) => (
          <li key={i}>{t}</li>
        ))}
      </ol>
    </div>
  );
};

export default TopTests;
