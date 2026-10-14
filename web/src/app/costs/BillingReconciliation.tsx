// Pythia / Copyright (c) 2025 Kevin Wyjad

export type BillingRow = {
  month: string;
  provider: string;
  ledger_usd: number | null;
  billed_usd: number | null;
  diff_usd: number | null;
  diff_pct: number | null;
  status: string;
  reason: string | null;
  checked_at: string | null;
};

export type BillingResponse = { has_table: boolean; rows: BillingRow[] };

const usd = (v: number | null) => (v === null || v === undefined ? "—" : `$${v.toFixed(2)}`);

const statusClass = (status: string) =>
  status === "mismatch"
    ? "text-red-700 font-semibold"
    : status === "ok"
      ? "text-green-700"
      : "text-fred-text";

// What each provider billed for a month beside what Fred's own cost ledger
// recorded (scripts/ci/billing_reconciliation.py, run on the 1st). A gap
// above 15% means a price, a call path or a model swap is being costed wrong.
export default function BillingReconciliation({ data }: { data: BillingResponse }) {
  return (
    <section className="space-y-2">
      <h2 className="text-xl font-semibold">Billed against recorded</h2>
      <p className="text-sm text-fred-text">
        Each month, what Anthropic and OpenAI report they billed, beside the cost Fred recorded
        for the same month. A provider more than 15% off is marked in red. Google publishes no
        cost report that this check can read.
      </p>
      {!data.has_table || data.rows.length === 0 ? (
        <p className="text-sm text-fred-text">No reconciliation has run yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left">
                <th className="px-2 py-1">Month</th>
                <th className="px-2 py-1">Provider</th>
                <th className="px-2 py-1 text-right">Recorded</th>
                <th className="px-2 py-1 text-right">Billed</th>
                <th className="px-2 py-1 text-right">Difference</th>
                <th className="px-2 py-1">Status</th>
                <th className="px-2 py-1">Note</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <tr key={`${r.month}-${r.provider}`} className="border-t border-fred-border">
                  <td className="px-2 py-1">{r.month}</td>
                  <td className="px-2 py-1">{r.provider}</td>
                  <td className="px-2 py-1 text-right">{usd(r.ledger_usd)}</td>
                  <td className="px-2 py-1 text-right">{usd(r.billed_usd)}</td>
                  <td className="px-2 py-1 text-right">
                    {r.diff_pct === null ? "—" : `${r.diff_pct > 0 ? "+" : ""}${r.diff_pct.toFixed(1)}%`}
                  </td>
                  <td className={`px-2 py-1 ${statusClass(r.status)}`}>{r.status}</td>
                  <td className="px-2 py-1">{r.reason ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
