import { ReactNode } from "react";

type KpiCardProps = {
  label: string;
  value: ReactNode;
};

const KpiCard = ({ label, value }: KpiCardProps) => {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
    </div>
  );
};

export default KpiCard;
