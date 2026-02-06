export default function SectionWrapper({ title, children }) {
  return (
    <div className="bg-slate-800 rounded-xl p-6 mb-6 shadow">
      <h2 className="text-xl font-semibold mb-4 text-blue-400">
        {title}
      </h2>
      {children}
    </div>
  )
}
