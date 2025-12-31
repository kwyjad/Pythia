import { ReactNode } from "react";

type KpiCardProps = {
  label: string;
  value: ReactNode;
};

const KpiCard = ({ label, value }: KpiCardProps) => {
  return (
    <div className="rounded-lg border border-fred-secondary bg-fred-surface p-4 shadow-fredCard">
      <div className="text-xs uppercase tracking-wide text-fred-muted">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-fred-primary">{value}</div>
    </div>
  );
};

export default KpiCard;
