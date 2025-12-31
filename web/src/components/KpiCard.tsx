import { ReactNode } from "react";

type KpiCardProps = {
  label: string;
  value: ReactNode;
};

const KpiCard = ({ label, value }: KpiCardProps) => {
  return (
    <div className="rounded-lg border border-fredBorder bg-fredSurface p-4 shadow-fredCard">
      <div className="text-xs uppercase tracking-wide text-fredMuted">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-fredPrimary">{value}</div>
    </div>
  );
};

export default KpiCard;
