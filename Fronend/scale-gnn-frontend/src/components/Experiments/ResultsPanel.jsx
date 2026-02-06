import React, { useEffect, useState } from "react"
import FullResults from "../Results/FullResults"
import TopTests from "../Results/TopTests"
import Metrics from "../Results/Metrics"
import Reduction from "../Results/Reduction"
import { fetchFullResults, fetchMetrics, fetchReduction } from "../../api/resultsApi"

export default function ResultsPanel({ experimentId }) {
  const [fullResults, setFullResults] = useState(null)
  const [topTests, setTopTests] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [reduction, setReduction] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!experimentId) return
    setLoading(true)

    const loadAll = async () => {
      try {
        const [fullRes, met, red] = await Promise.all([
          fetchFullResults(experimentId),
          fetchMetrics(experimentId),
          fetchReduction(experimentId),
        ])

        setFullResults(fullRes)
        setMetrics(met)
        setReduction(red)

        // pick top 10 based on risk_score
        if (fullRes.ranking && fullRes.ranking.length > 0) {
          const sorted = [...fullRes.ranking].sort(
            (a, b) => b.risk_score - a.risk_score
          )
          setTopTests(sorted.slice(0, 10))
        } else {
          setTopTests([])
        }

      } catch (err) {
        console.error("Error loading results:", err)
      } finally {
        setLoading(false)
      }
    }

    loadAll()
  }, [experimentId])

  if (loading) return <p className="text-slate-400">Loading results...</p>

  return (
    <div className="space-y-6">
      {reduction && <Reduction data={reduction} />}
      {metrics && <Metrics data={metrics} />}
      {topTests.length > 0 && <TopTests data={topTests} />}
      {fullResults && <FullResults data={fullResults} />}
    </div>
  )
}
