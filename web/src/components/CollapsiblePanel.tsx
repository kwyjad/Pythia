import type { ReactNode } from "react";

type CollapsiblePanelProps = {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
};

const CollapsiblePanel = ({
  title,
  children,
  defaultOpen = false
}: CollapsiblePanelProps) => {
  return (
    <details
      className="rounded-lg border border-slate-800 bg-slate-900/60"
      open={defaultOpen}
    >
      <summary className="cursor-pointer select-none px-4 py-2 text-sm font-semibold text-slate-100">
        {title}
      </summary>
      <div className="border-t border-slate-800 px-4 py-3">{children}</div>
    </details>
  );
};

export default CollapsiblePanel;
