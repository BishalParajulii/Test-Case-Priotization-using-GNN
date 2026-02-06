import { useEffect, useState } from "react"
import { fetchRanking } from "../../api/resultsApi"
import SectionWrapper from "../Layout/SectionWrapper"

export default function Ranking({ experimentId }) {
  const [data, setData] = useState([])

  useEffect(() => {
    fetchRanking(experimentId).then(setData)
  }, [experimentId])

  return (
    <SectionWrapper title="Test Case Ranking">
      <table className="w-full text-left">
        <thead className="text-slate-400">
          <tr>
            <th>#</th>
            <th>Test Case</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="border-t border-slate-700">
              <td>{i + 1}</td>
              <td>{row.test}</td>
              <td>{row.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </SectionWrapper>
  )
}
