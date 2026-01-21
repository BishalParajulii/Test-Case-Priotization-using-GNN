const ReportButton = ({ experimentId }) => {
  const downloadReport = () => {
    window.open(
      `http://127.0.0.1:8000/reports/${experimentId}`,
      "_blank"
    );
  };

  return (
    <button
      onClick={downloadReport}
      style={{
        marginTop: "20px",
        padding: "10px 16px",
        background: "#2563eb",
        color: "#fff",
        border: "none",
        borderRadius: "6px",
        cursor: "pointer",
      }}
    >
      📄 Download Experiment Report
    </button>
  );
};

export default ReportButton;
