export default function TabsHeader({ tabs, active, setActive }) {
  return (
    <div className="flex gap-4 mb-6">
      {tabs.map(tab => (
        <button
          key={tab.key}
          onClick={() => setActive(tab.key)}
          className={`px-4 py-2 rounded-lg ${
            active === tab.key
              ? "bg-blue-600"
              : "bg-slate-700 hover:bg-slate-600"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}
