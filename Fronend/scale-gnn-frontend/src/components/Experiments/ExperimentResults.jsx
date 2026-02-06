import { useState } from "react"
import TabsHeader from "../Tabs/TabsHeader"
import Ranking from "../Results/Ranking"
import Metrics from "../Results/Metrics"
import Reduction from "../Results/Reduction"
import TopTests from "../Results/TopTests"
import FullResults from "../Results/FullResults"

export default function ExperimentResults({ experimentId }) {
  const [active, setActive] = useState("overview")

  const tabs = [
    { key: "overview", label: "Overview" },
    { key: "ranking", label: "Ranking" },
    { key: "top", label: "Top Tests" },
    { key: "metrics", label: "Metrics" },
    { key: "reduction", label: "Reduction" },
  ]

  return (
    <>
      <TabsHeader tabs={tabs} active={active} setActive={setActive} />

      {active === "overview" && <FullResults experimentId={experimentId} />}
      {active === "ranking" && <Ranking experimentId={experimentId} />}
      {active === "top" && <TopTests experimentId={experimentId} />}
      {active === "metrics" && <Metrics experimentId={experimentId} />}
      {active === "reduction" && <Reduction experimentId={experimentId} />}
    </>
  )
}
